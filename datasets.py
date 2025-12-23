import numpy as np
from multiprocessing import Pool
from numpy.lib.shape_base import _take_along_axis_dispatcher
import torch
from torch.nn.functional import feature_alpha_dropout
import torch.utils.data as D
import matplotlib.pyplot as plt


'''数据工具函数与数据集定义'''
def _load_signal(p):
    return np.load(p).astype(np.float32), p


def load_signal_cache(paths, 
                      cache_limit=10, # in GB
                      n_jobs=1):
    """预加载波形到内存缓存
    
    参数：
        paths: 波形文件路径列表
        cache_limit: 内存上限（GB），超过即提前返回（避免 OOM）
        n_jobs: 进程数，大于1则多进程并行加载
    返回：
        cache: {路径: numpy.ndarray(float32)}，若超限则提前返回已加载部分
    """

    size_in_gb = 0
    cache = {}
    # 多进程并行加载
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for s, p in pool.imap_unordered(_load_signal, paths):
                cache[p] = s  # 保存到缓存
                size_in_gb += s.nbytes / (1024 ** 3)  # 累计占用（GB）
                # 超出内存上限则提前返回
                if size_in_gb > cache_limit:
                    print(f'{len(cache)} items / {size_in_gb:.2f} GB cache loaded.')
                    return cache
    else:
        # 单进程顺序加载
        for p in paths:
            s, _ = _load_signal(p)
            size_in_gb += s.nbytes / (1024 ** 3)
            if size_in_gb > cache_limit:
                print(f'{len(cache)} items / {size_in_gb:.2f} GB cache loaded.')
                return cache

    print(f'{len(cache)} items / {size_in_gb:.2f} GB cache loaded.')
    return cache


'''
Dataset
'''
class G2NetDataset(D.Dataset):
    '''G2Net 数据集
    
    支持：
    - 可选缓存（cache/test_cache）  test_cache好像是多余的
    - mixup（随机/只与负样本）
    - 伪标签拼接（pseudo_label）
    - 双路 transforms（transforms / transforms2）
    - 返回索引/测试标记
    
        Amplitude stats
    [RAW]
    max of max: [4.6152116e-20, 4.2303907e-20, 1.1161064e-20]
    mean of max: [1.8438003e-20, 1.8434544e-20, 5.0978556e-21]
    max of mean: [1.5429503e-20, 1.5225015e-20, 3.1584522e-21]
    [BANDPASS]
    max of max: [1.7882743e-20, 1.8305723e-20, 9.5750025e-21]
    mean of max: [7.2184587e-21, 7.2223450e-21, 2.4932809e-21]
    max of mean: [6.6964011e-21, 6.4522511e-21, 1.4383649e-21]
    '''
    def __init__(self, 
                 paths, 
                 targets=None, 
                 spectrogram=None, 
                 norm_factor=None,
                 transforms=None,
                 transforms2=None, 
                 cache=None,
                 mixup=False,
                 mixup_alpha=0.4, 
                 mixup_option='random',
                 hard_label=False,
                 lor_label=False,
                 is_test=False,
                 pseudo_label=False,
                 test_targets=None,
                 test_paths=None,
                 test_cache=None,
                 return_index=False,
                 return_test_index=False,
                 ):
        """初始化数据集并可选拼接伪标签数据
        
        参数仅列要点：
            paths/targets: 训练/验证的数据路径与标签
            spectrogram: 可选频谱生成器（已废弃）
            norm_factor: 每通道归一化因子（已废弃）
            transforms/transforms2: 主/副路数据增强
            cache/test_cache: 预加载缓存（路径->numpy）
            mixup/mixup_alpha/mixup_option: mixup 开关、系数、策略
            hard_label/lor_label: mixup 后标签二值化或逻辑或
            is_test: 测试模式（禁用 mixup / pseudo_label）
            pseudo_label + test_*: 启用伪标签时拼接测试集并标记来源
            return_index/return_test_index: 是否返回样本索引/伪标签标记
        """
        self.paths = paths
        self.targets = targets
        self.negative_idx = np.where(self.targets == 0)[0]
        self.spectr = spectrogram
        self.norm_factor = norm_factor
        self.transforms = transforms
        self.transforms2 = transforms2 # additional transforms
        self.cache = cache
        #
        # print('cache', cache)
        
        self.test_cache = None
        self.mixup = mixup
        self.alpha = mixup_alpha
        self.mixup_option = mixup_option
        self.hard_label = hard_label
        self.lor_label = lor_label
        self.is_test = is_test
        self.pseudo_label = pseudo_label
        self.return_index = return_index
        self.return_test_index = return_test_index
        if self.is_test:
            self.mixup = False
            self.pseudo_label = False
        if self.pseudo_label:
            # 训练集与伪标签测试集拼接
            self.paths = np.concatenate([self.paths, test_paths])
            self.targets = np.concatenate([self.targets, test_targets])
            self.test_cache = test_cache
            self.test_index = np.array([0]*len(paths) + [1]*len(test_paths)).astype(np.uint8)
        else:
            self.test_index = np.array([0]*len(self.paths)).astype(np.uint8)

    def __len__(self):
        """返回数据集大小"""
        return len(self.paths)
    
    def __getitem__(self, index):
        """读取单个样本，执行 mixup（若开启），并按需要附加索引标记
        
        处理流程：
        1) 取样本：调用 _get_signal_target(index) 得到 signal/sub_signal/target
        2) 可选 mixup：
           - 选第二个样本：random 任意样本；negative 仅负样本（噪声弱化 lam=max(lam,1-lam)）
           - lam ~ Beta(alpha, alpha)，按 lam 对主/副路信号线性混合
           - 标签：
             * lor_label=True 时逻辑或：t = t1 + t2 - t1*t2
             * 否则按 lam 插值，若 hard_label 给定阈值再二值化
        3) 组装输出：
           - 无副路 => [signal, target]；有副路 => [signal, sub_signal, target]
           - return_index=True 追加样本索引
           - return_test_index=True 追加 test_index（伪标签来源标记 0/1）
        返回 tuple
        """
        signal, sub_signal, target = self._get_signal_target(index)
        if self.mixup:
            # mixup 支持随机或仅与负样本混合
            if self.mixup_option == 'random':
                idx2 = np.random.randint(0, len(self))
                lam = np.random.beta(self.alpha, self.alpha)
            elif self.mixup_option == 'negative':
                idx2 = np.random.randint(0, len(self.negative_idx))
                idx2 = self.negative_idx[idx2]
                lam = np.random.beta(self.alpha, self.alpha)
                lam = max(lam, 1-lam) # negative noise is always weaker
            signal2, sub_signal2, target2 = self._get_signal_target(idx2)
            signal = lam * signal + (1 - lam) * signal2
            if sub_signal is not None:
                sub_signal = lam * sub_signal + (1 - lam) * sub_signal2
            if self.lor_label:
                target = target + target2 - target * target2
            else:
                target = lam * target + (1 - lam) * target2
                if self.hard_label:
                    target = (target > self.hard_label).float()
        if sub_signal is None:
            outputs = [signal, target]
        else:
            outputs = [signal, sub_signal, target]
        if self.return_index:
            outputs.append(torch.tensor(index))
        if self.return_test_index:
            outputs.append(torch.tensor(self.test_index[index]))
        return tuple(outputs)
        
    def _get_signal_target(self, index):
        """加载波形、应用变换并返回 (signal1, signal2, target)
        
        流程：
        1) 读取波形（优先缓存）：path = self.paths[index]；先查 cache，再查 test_cache，均无则 np.load(float32)
        2) 可选归一化（已废弃）：若 norm_factor 存在，对每通道除以对应因子
        3) 数据增强与双路输出：transforms 作用于 signal.copy() 得到 signal1；transforms2 作用于原始 signal 得到 signal2（可为 None）
        4) 转 Tensor（已废弃标记）：若 signal1/2 为 ndarray，则转 torch.float32
        5) 可选频谱生成（已废弃）：若 spectr 存在，对 signal1 做频谱变换
        6) 生成标签：若 targets 提供，取对应标签 unsqueeze(0) 为 float32；否则返回 0 的占位张量 shape=(1,)
        7) 返回 (signal1, signal2, target)，signal2 可能为 None
        """
        path = self.paths[index]
        if self.cache is not None and path in self.cache.keys():
            signal = self.cache[path].copy()
        elif self.test_cache is not None and path in self.test_cache.keys():
            signal = self.test_cache[path].copy()
        else:
            signal = np.load(path).astype(np.float32)

        if self.norm_factor is not None: # DEPRECATED: normalization
            for ch in range(3):
                signal[ch] = signal[ch] / self.norm_factor[ch]

        if self.transforms is not None:
            signal1 = self.transforms(signal.copy())
        else:
            signal1 = signal
        if self.transforms2 is not None:
            signal2 = self.transforms2(signal)
        else:
            signal2 = None

        if isinstance(signal1, np.ndarray): # DEPRECATED: to tensor
            signal1 = torch.from_numpy(signal1).float()
        if signal2 is not None and isinstance(signal2, np.ndarray): # DEPRECATED: to tensor
            signal2 = torch.from_numpy(signal2).float()
        
        if self.spectr is not None: # DEPRECATED: spectrogram generation
            signal1 = self.spectr(signal1)
            
        if self.targets is not None:
            target = torch.tensor(self.targets[index]).unsqueeze(0).float()
        else:
            target = torch.tensor(0).unsqueeze(0).float()
        return signal1, signal2, target
