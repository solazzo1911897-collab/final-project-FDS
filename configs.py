"""
Kaggle G2Net 引力波检测竞赛 - 模型训练配置文件

本文件包含所有实验的配置类，用于定义模型架构、训练参数、数据增强策略等。
每个配置类继承自基础配置类，通过修改特定参数来创建不同的实验变体。

主要配置类型：
1. Baseline: 基础配置，使用CQT时频变换和EfficientNet
2. Nspec系列: 使用连续小波变换(CWT)的配置
3. Seq系列: 使用1D序列模型（ResNet1d, DenseNet1d, WaveNet1d）的配置
4. Pseudo系列: 使用伪标签训练的配置
5. MultiInstance: 多实例学习配置
"""

from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from nnAudio.Spectrogram import CQT  # 常数Q变换（时频分析）
from cwt_pytorch import ComplexMorletCWT  # 复Morlet小波变换

from utils import (
    EarlyStopping, SaveSnapshot) 
from utils import TrainHook

from datasets import G2NetDataset
from architectures import SpectroCNN
# from models1d_pytorch import *
# from loss_functions import BCEWithLogitsLoss
from metrics import AUC
from transforms import *


# 数据目录配置
INPUT_DIR = Path('input/').expanduser()


class Baseline:
    """
    基础配置类 - 所有其他配置类的父类
    
    使用CQT（常数Q变换）将时域信号转换为时频图，然后用EfficientNet-B7进行图像分类。
    这是最简单的baseline配置，没有数据增强。
    """
    name = 'baseline'  # 实验名称
    seed = 2021  # 随机种子，确保可复现性
    train_path = INPUT_DIR/'train.csv'  # 训练集CSV路径
    test_path = INPUT_DIR/'test.csv'   # 测试集CSV路径
    train_cache = INPUT_DIR/'train_cache.pickle' # 训练集缓存路径（可选，用于加速数据加载）
    test_cache = INPUT_DIR/'test_cache.pickle'   # 测试集缓存路径（可选）
    
    # 交叉验证配置
    cv = 2  # 5折交叉验证
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)  # 分层K折，保持类别分布
    
    # 数据集配置
    dataset = G2NetDataset  # 数据集类
    dataset_params = dict()  # 数据集额外参数

    # 模型配置
    model = SpectroCNN  # 频谱图CNN模型（将时域信号转为频谱图后分类）
    model_params = dict(
        model_name='tf_efficientnet_b7',  # TensorFlow版本的EfficientNet-B7（ImageNet预训练）
        pretrained=True,  # 使用预训练权重
        num_classes=1,  # 二分类任务（输出1个值）
        spectrogram=CQT,  # 使用CQT（常数Q变换）作为时频变换方法
        spec_params=dict(
            sr=2048,      # 采样率：2048 Hz（引力波数据采样率）
            fmin=20,      # 最小频率：20 Hz
            fmax=1024,    # 最大频率：1024 Hz
            hop_length=64 # 跳跃长度：64（控制时间分辨率）
        ),
    )
    weight_path = None  # 预训练权重路径（None表示从头训练）
    
    # 训练超参数
    num_epochs = 5  # 训练轮数
    batch_size = 64  # 批次大小  T4 16GB 64大概占满
    # 128 T4 RuntimeError: CUDA out of memory. Tried to allocate 144.00 MiB (GPU 0; 14.74 GiB total capacity; 13.38 GiB already allocated; 64.12 MiB free; 13.59 GiB reserved in total by PyTorch)
    # 128 A100 40g 显存占29g 
    
    # 优化器配置
    optimizer = optim.Adam  # Adam优化器
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)  # 学习率2e-4，权重衰减1e-6
    
    # 学习率调度器配置
    scheduler = CosineAnnealingWarmRestarts  # 余弦退火带重启
    # 参数说明（当前配置：T_0=5, T_mult=1, eta_min=1e-6）：
    # - T_0=5：首个周期长度 5 个 epoch，LR 在 5 个 epoch 内从初始值余弦下降到 eta_min
    # - T_mult=1：周期倍增因子；=1 表示每个周期都保持 5（如设 2 则 5→10→20…）
    # - eta_min=1e-6：每个周期的最低学习率，不降到 0
    # - initial_lr：由优化器设置（如 Adam lr=2e-4），余弦在 [initial_lr, eta_min] 之间摆动
    # 工作方式：每个周期内 LR 按余弦平滑下降到 eta_min，周期末瞬间重启回初始 LR，再进入下个周期。
    # 这种周期性“脉冲”有助于跳出局部最优；因 T_mult=1，当前设置每 5 个 epoch 重启一次。
    # 工作方式：LR 下降到 0 后，突然瞬间跳回最大值，然后再次下降。就像周期性的“脉冲”。 目的：那个瞬间跳回（Restart）是为了把模型从局部最优解中“踢”出来，让它去寻找更好的坑。
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)  # 初始周期5，最小学习率1e-6
    scheduler_target = None  # 调度器监控指标（None表示监控loss）
    batch_scheduler = False  # 是否每个batch更新学习率（False表示每个epoch更新）
    
    # 损失函数和评估指标
    criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失（带sigmoid）
    eval_metric = AUC().torch  # 评估指标：AUC（ROC曲线下面积）
    monitor_metrics = []  # 额外监控指标列表
    
    # 训练技术配置
    amp = True  # 自动混合精度训练（FP16），加速训练并节省显存
    # parallel = None  # 并行训练配置（None表示单GPU）
    deterministic = False  # 是否使用确定性算法（False表示允许非确定性操作以提升性能）
    # clip_grad = 'value'  # 梯度裁剪方式：'value'表示按值裁剪
    # max_grad_norm = 10000  # 梯度裁剪阈值（非常大，实际不裁剪）
    
    # 训练钩子和回调函数
    hook = TrainHook()  # 训练钩子
    callbacks = [
        EarlyStopping(patience=5, maximize=True),  # 早停：5个epoch无提升则停止，监控指标越大越好
        SaveSnapshot()  # 保存模型快照
    ]

    # 数据增强配置
    transforms = dict(
        train=None,  # 训练集数据增强（None表示无增强）
        test=None,   # 测试集数据增强
        tta=None     # 测试时增强（Test Time Augmentation）
    )

    # 伪标签配置
    # pseudo_labels = None  # 伪标签路径（None表示不使用伪标签）
    debug = False  # 调试模式（False表示正常训练）


class B2CQT(Baseline):
    """
    改进的基础配置 - 添加了数据增强和图像尺寸调整
    
    主要改进：
    - 使用更小的EfficientNet-B2（相比B7更快）
    - 添加高斯噪声增强（SNR 15-30 dB）
    - 将频谱图调整到256x512尺寸
    - 使用更小的hop_length=8（更高的时间分辨率）
    """
    name = 'b2_cqt'
    model_params = dict(
        model_name='tf_efficientnet_b2',  # 更小的模型，训练更快
        pretrained=True,
        num_classes=1,
        spectrogram=CQT,
        spec_params=dict(
            sr=2048,      # 采样率：2048 Hz（引力波数据采样率）
            fmin=20,      # 最小频率：20 Hz
            fmax=1024,    # 最大频率：1024 Hz
            hop_length=64 # 跳跃长度：64（控制时间分辨率）
        ),
        # resize_img=(256, 512),  # 将频谱图调整到256x512
        # upsample='bicubic'  # 使用双三次插值上采样 用于resize
    )
    transforms = dict(
        train=Compose([
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.25)  # 25%概率添加高斯噪声
        ]),
        test=None,
        tta=None
    )
    dataset_params = dict(
        norm_factor=[4.61e-20, 4.23e-20, 1.11e-20]  # 三个探测器的归一化因子
    )
    num_epochs = 8  # 增加训练轮数
    scheduler_params = dict(T_0=8, T_mult=1, eta_min=1e-6)  # 调整调度器周期
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)  # 提高学习率到1e-3


class B2CWT(B2CQT): 
    """
    使用CWT的高性能配置 - B2CWT
    
    主要特点：
    - 可训练的小波宽度（trainable_width=True），允许模型学习最优小波参数
    - 使用GeM池化（Generalized Mean Pooling）替代平均池化
    - 更高的时间分辨率（stride=4）
    - 图像尺寸128x1024（更宽的时间维度）
    
    
        model_params = dict(
        model_name='tf_efficientnet_b2',  # 更小的模型，训练更快
        pretrained=True,
        num_classes=1,
        spectrogram=CQT,
        spec_params=dict(
            sr=2048, 
            fmin=16,      # 降低最小频率到16 Hz
            fmax=1024, 
            hop_length=8  # 更小的跳跃长度，提高时间分辨率
        ),
        resize_img=(256, 512),  # 将频谱图调整到256x512
        upsample='bicubic'  # 使用双三次插值上采样
    )
    """
    name = 'b2_cwt'
    model_params = dict(
        model_name='tf_efficientnet_b2',  # tf_efficientnet_b0  # b7爆显存 换用80GB的a100 80G还是不行
        pretrained=True,
        num_classes=1,
        spectrogram=ComplexMorletCWT,
        spec_params=dict(
            fs=2048, 
            lower_freq=16, 
            upper_freq=1024, 
            wavelet_width=8,      # 更大的小波宽度
            trainable_width=True, # 可训练的小波宽度（端到端优化）
            stride=4,             # 更小的步长，提高时间分辨率
            n_scales=128          # 128个尺度
        ),
        resize_img=(128, 1024),   # 更宽的时间维度（1024）
        # custom_classifier='gem',  # GeM池化（Generalized Mean Pooling） 自己实现的模块
        upsample='bicubic'
    )
    
    # model_params = dict(
    #     model_name='tf_efficientnet_b2',  # 更小的模型，训练更快
    #     pretrained=True,
    #     num_classes=1,
    #     spectrogram=CQT,
    #     spec_params=dict(
    #         sr=2048, 
    #         fmin=16,      # 降低最小频率到16 Hz
    #         fmax=1024, 
    #         hop_length=8  # 更小的跳跃长度，提高时间分辨率
    #     ),
    #     resize_img=(256, 512),  # 将频谱图调整到256x512
    #     upsample='bicubic'  # 使用双三次插值上采样
    # )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),  # 归一化三个探测器
            BandPass(lower=12, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=12, upper=512),
        ])
    )
    dataset_params = dict()