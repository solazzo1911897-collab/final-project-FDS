import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn.parameter import Parameter


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 
            (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return f'GeM(p={self.p}, eps={self.eps})'

class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x): 
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)


class SpectroCNN(nn.Module):

    def __init__(self, 
                 model_name='efficientnet_b7', 
                 pretrained=False, 
                 num_classes=1,
                 timm_params={}, 
                 spectrogram=None,
                 spec_params={},
                 resize_img=None,
                 upsample='nearest', 
                 # mixup='mixup',
                 norm_spec=False,
                 return_spec=False):
        
        super().__init__()
        if isinstance(spectrogram, nn.Module): # deprecated
            self.spectrogram = spectrogram
        else:
            self.spectrogram = spectrogram(**spec_params)
        self.is_cnnspec = self.spectrogram.__class__.__name__ in [
            'WaveNetSpectrogram', 'CNNSpectrogram', 'MultiSpectrogram', 'ResNetSpectrogram']

        self.cnn = timm.create_model(model_name, 
                                     pretrained=pretrained, 
                                     num_classes=num_classes,
                                     **timm_params)
        # self.mixup_mode = 'input'
                # 如果指定了自定义分类器或注意力机制，需要替换 CNN backbone 的原始分类头
        # if custom_classifier != 'none' or custom_attention != 'none':
        model_type = self.cnn.__class__.__name__  # 获取模型类型名称（如 'EfficientNet'）
        try:
            # 获取原始分类器的输入特征维度（用于后续构建自定义分类器）
            feature_dim = self.cnn.get_classifier().in_features
            # 移除原始分类器（重置为空的分类器，只保留特征提取部分）
            self.cnn.reset_classifier(0, '')
        except:
            # 如果模型不支持 get_classifier() 方法，抛出错误
            raise ValueError(f'Unsupported model type: {model_type}')

        
        # GeM（Generalized Mean Pooling）：可学习的池化方式，比平均池化更灵活
        # p=3 表示使用 L3 范数，eps=1e-4 是数值稳定性参数
        # global_pool = GeM(p=3, eps=1e-4)
        global_pool = nn.Identity()
        

        self.cnn = nn.Sequential(
            self.cnn, 
            global_pool, 
            Flatten(),
            nn.Linear(feature_dim, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, num_classes)
        )
        
        
        
        
        self.norm_spec = norm_spec
        if self.norm_spec:
            self.norm = nn.BatchNorm2d(3)
        self.resize_img = resize_img
        if isinstance(self.resize_img, int):
            self.resize_img = (self.resize_img, self.resize_img)
        self.return_spec = return_spec
        self.upsample = upsample
        # self.mixup = mixup
        # assert self.mixup in ['mixup', 'cutmix']
    
    def feature_mode(self):
        self.cnn[-1] = nn.Identity()
        self.cnn[-2] = nn.Identity()

    def forward(self, s, lam=None, idx=None): # s: (batch size, wave channel, length of wave)  # lam, idx 参数保留以保持接口兼容性，但不再使用
        # ========== 1. 获取输入信号维度 ==========
        # s: 输入波形信号，形状为 (batch_size, wave_channel, wave_length)
        # 例如：(32, 3, 4096) 表示 32 个样本，3 个通道（3 个探测器），每个信号长度 4096
        bs, ch, w = s.shape
        
        # ========== 2. 生成时频图（Spectrogram） ==========

        # 使用传统时频图生成方法（如 MelSpectrogram, CQT 等）
        # 这些方法通常只处理单通道输入，需要先将多通道展平
        s = s.view(bs * ch, w)  # 将 (batch, channels, length) 展平为 (batch*channels, length)
        
        # 禁用自动混合精度（AMP），因为 MelSpectrogram 在使用 AMP 时可能导致 NaN
        with torch.cuda.amp.autocast(enabled=False): 
            # 生成时频图，输出形状为 (batch_size * wave_channel, freq, time)
            spec = self.spectrogram(s)
        
        # 获取时频图的频率和时间维度
        _, f, t = spec.shape
        # 重新组织维度：将 (batch*channels, freq, time) 恢复为 (batch, channels, freq, time)
        spec = spec.view(bs, ch, f, t)
        
        # ========== 3. 应用 Mixup/CutMix 数据增强 ==========
        # lam: mixup 的混合系数（lambda），如果为 None 则不进行 mixup
        # idx: 用于 mixup 的样本索引
        # if lam is not None: 
        #     if self.mixup == 'mixup' and self.mixup_mode == 'input':
        #         # 在输入层（时频图）应用 Mixup：混合两个样本的时频图
        #         # 公式：spec_mixed = lam * spec[idx1] + (1-lam) * spec[idx2]
        #         spec, lam = mixup(spec, lam, idx)
        #     elif self.mixup == 'cutmix':
        #         # 应用 CutMix：用另一个样本的矩形区域替换当前样本的对应区域
        #         spec, lam = cutmix(spec, lam, idx)
        

        # ========== 调整时频图尺寸 ==========
        # 将时频图调整为指定尺寸，确保输入到 CNN 的尺寸一致
        # 
        # 为什么需要 resize：
        # 1. 统一输入尺寸：不同配置可能产生不同尺寸的时频图（如 n_scales、stride 不同）
        #    通过 resize 统一尺寸，便于模型处理和批处理
        # 2. 适配 CNN 架构：某些 CNN 架构（如 EfficientNet）对输入尺寸有特定要求
        #    统一尺寸可以更好地利用预训练权重
        # 3. 控制计算量：通过调整尺寸可以平衡精度和计算效率
        #    例如 (128, 1024) 比 (256, 2048) 计算量小 4 倍
        #
        # 参数说明：
        # - spec: 时频图，形状为 [batch, channels, freq, time]
        #   例如：[32, 3, 128, 1024] 或 [32, 3, 256, 512]
        # - size=self.resize_img: 目标尺寸，例如 (128, 1024) 表示 (高度, 宽度)
        #   - 高度对应频率维度（freq/n_scales）
        #   - 宽度对应时间维度（time）
        # - mode=self.upsample: 插值模式，例如 'bicubic'、'bilinear'、'nearest'
        #   - 'bicubic': 双三次插值，质量最好但较慢
        #   - 'bilinear': 双线性插值，平衡质量和速度
        #   - 'nearest': 最近邻插值，最快但质量较低
        #
        # 输出：
        # - spec: 调整后的时频图，形状为 [batch, channels, resize_img[0], resize_img[1]]
        #   例如：输入 [32, 3, 256, 512]，resize_img=(128, 1024)
        #   输出 [32, 3, 128, 1024]
        if self.resize_img is not None:
            spec = F.interpolate(spec, size=self.resize_img, mode=self.upsample)

        if self.norm_spec:
            spec = self.norm(spec)
        
        # if self.mixup_mode == 'manifold':
        #     self.cnn[3][1].update(lam, idx)

        # if self.return_spec and lam is not None:
        #     return self.cnn(spec), spec, lam
        if self.return_spec:
            return self.cnn(spec), spec
        # elif lam is not None:
        #     return self.cnn(spec), lam
        else:
            return self.cnn(spec)