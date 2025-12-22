"""
数据准备脚本 - prep_data.py

本脚本的主要功能：
1. 读取 Kaggle 竞赛的原始数据文件（training_labels.csv 和 sample_submission.csv）
2. 为每个样本生成对应的波形文件路径（根据 ID 的前三个字符组织目录结构）
3. 生成新的 train.csv 和 test.csv 文件，包含 id、target（仅训练集）和 path 列
4. 可选：生成波形缓存文件（pickle 格式），用于加速训练时的数据加载

使用示例：
    python prep_data.py                          # 仅生成 CSV 文件
    python prep_data.py --cache                  # 同时生成缓存文件
"""

import argparse
from pathlib import Path
import pandas as pd
import pickle
import gc
import numpy as np
from multiprocessing import Pool


def _load_signal(p):
    """加载单个信号文件"""
    return np.load(p).astype(np.float32), p


def load_signal_cache(paths, cache_limit=10, n_jobs=4):
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
            cache[p] = s
            size_in_gb += s.nbytes / (1024 ** 3)
            if size_in_gb > cache_limit:
                print(f'{len(cache)} items / {size_in_gb:.2f} GB cache loaded.')
                return cache

    print(f'{len(cache)} items / {size_in_gb:.2f} GB cache loaded.')
    return cache


if __name__ == "__main__":
    # ========== 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description='准备 G2Net 竞赛数据文件')
    parser.add_argument("--root_dir", type=str, default='input',
                        help="原始数据目录（包含 training_labels.csv 和 sample_submission.csv）")
    parser.add_argument("--export_dir", type=str, default='input',
                        help="输出目录（生成的 train.csv、test.csv 和缓存文件将保存在这里）")
    parser.add_argument("--cache", action='store_true', 
                        help="是否生成波形缓存文件（需要至少 32GB RAM，可显著加速训练）")
   
    opt = parser.parse_args()

    # ========== 固定硬件配置参数 ==========
    # 固定参数：CPU核心数、RAM大小(GB)
    N_CPU = 4  # 并行加载的进程数
    N_RAM = 32  # RAM大小（GB），用于计算缓存限制
    if opt.cache:
        # 缓存大小限制为 RAM 的一半，避免内存溢出
        cache_limit = N_RAM // 2
        print(f'最大缓存大小设置为 {cache_limit} GB')

    # ========== 设置目录路径 ==========
    root_dir = Path(opt.root_dir).expanduser()  # 展开用户目录路径（如 ~/data -> /home/user/data）
    
    # ========== 读取原始数据文件 ==========
    train = pd.read_csv(root_dir/'training_labels.csv')  # 训练集标签（包含 id 和 target 列）
    test = pd.read_csv(root_dir/'sample_submission.csv')  # 测试集提交模板（包含 id 和 target 列）
    
    export_dir = Path(opt.export_dir).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录（如果不存在）

    # ========== 处理训练集 ==========
    print('===== TRAIN =====')
    # 根据 ID 的前三个字符创建三级目录结构
    # 例如：ID = "abc123def" -> train/a/b/c/abc123def.npy
    # 这种组织方式可以避免单个目录下文件过多，提高文件系统性能
    train['path'] = train['id'].apply(
        lambda x: root_dir/f'train/{x[0]}/{x[1]}/{x[2]}/{x}.npy'
    )
    # 检查文件是否存在，只保留存在的文件
    train = train[train['path'].apply(lambda x: Path(x).exists())]
    print(f'训练集文件数量: {len(train)} (已过滤不存在的文件)')
    # 保存包含路径信息的训练集 CSV 文件
    train.to_csv(export_dir/'train.csv', index=False)
    
    # ========== 可选：生成训练集波形缓存 ==========
    if opt.cache:
        # load_signal_cache 函数会：
        # 1. 并行加载所有波形文件（.npy 格式）到内存
        # 2. 限制总缓存大小不超过 cache_limit GB
        # 3. 返回一个字典：{文件路径: 波形数据数组}
        train_cache = load_signal_cache(
            train['path'].values,  # 所有训练集文件路径
            cache_limit,            # 缓存大小限制（GB）
            n_jobs=N_CPU           # 并行加载的进程数
        )
        # 将缓存保存为 pickle 文件，训练时可以直接加载到内存
        with open(export_dir/'train_cache.pickle', 'wb') as f:
            pickle.dump(train_cache, f)
        # 释放内存
        del train_cache
        gc.collect()

    # ========== 处理测试集 ==========
    print('===== TEST =====')
    # 同样根据 ID 的前三个字符创建目录结构
    test['path'] = test['id'].apply(
        lambda x: root_dir/f'test/{x[0]}/{x[1]}/{x[2]}/{x}.npy'
    )
    # 检查文件是否存在，只保留存在的文件
    test = test[test['path'].apply(lambda x: Path(x).exists())]
    print(f'测试集文件数量: {len(test)} (已过滤不存在的文件)')
    # 保存包含路径信息的测试集 CSV 文件
    test.to_csv(export_dir/'test.csv', index=False)
    
    # ========== 可选：生成测试集波形缓存 ==========
    if opt.cache:
        test_cache = load_signal_cache(
            test['path'].values,   # 所有测试集文件路径
            cache_limit,            # 缓存大小限制（GB）
            n_jobs=N_CPU           # 并行加载的进程数
        )
        # 将缓存保存为 pickle 文件
        with open(export_dir/'test_cache.pickle', 'wb') as f:
            pickle.dump(test_cache, f)
        # 释放内存
        del test_cache
        gc.collect()

