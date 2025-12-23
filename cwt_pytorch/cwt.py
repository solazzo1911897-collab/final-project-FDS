"""
连续小波变换（Continuous Wavelet Transform, CWT）的 PyTorch 实现

本模块提供了 GPU 加速的连续小波变换，用于将时域信号转换为时频表示（scalogram）。
小波变换能够同时提供时间和频率信息，特别适合分析非平稳信号（如引力波信号）。

主要类：
- ContinuousWaveletTransform: 基础 CWT 类，使用卷积实现
- ComplexMorletCWT: 复 Morlet 小波变换（适合分析瞬态信号）
- RickerCWT: Ricker 小波变换（墨西哥帽小波，适合检测信号边缘）

实现基于：
https://github.com/Kevin-McIsaac/cmorlet-tensorflow/blob/master/cwt.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousWaveletTransform(nn.Module):
    """
    GPU 加速的连续小波变换基础类
    
    连续小波变换（CWT）通过将信号与不同尺度的小波函数进行卷积，
    得到信号的时频表示（scalogram）。每个尺度对应不同的频率范围。
    
    实现方式：
    - 使用 2D 卷积（conv2d）实现小波变换
    - 将不同尺度的小波函数作为卷积核
    - 通过卷积操作计算信号与小波的相似度
    
    Args:
        n_scales: (int) 小波尺度数量，决定频率分辨率
            - 更多尺度 = 更高的频率分辨率，但计算量更大
            - 每个尺度对应一个频率范围
        border_crop: (int) 边界裁剪数量（默认 0）
            - 小波变换在信号边界处会产生边界效应（边界失真）
            - 此参数指定在计算 CWT 后从两端裁剪的样本数
            - 允许输入比最终输出更长的信号，以去除边界效应
        stride: (int) 卷积步长（默认 1）
            - 控制时间维度的下采样率
            - stride=1: 保持原始时间分辨率
            - stride>1: 降低时间分辨率，减少计算量
    """
    def __init__(self, n_scales, border_crop=0, stride=1):
        super().__init__()
        self.n_scales = n_scales      # 小波尺度数量
        self.border_crop = border_crop # 边界裁剪数量
        self.stride = stride           # 卷积步长
        self._build_wavelet_bank()     # 构建小波滤波器组（由子类实现）

    def _build_wavelet_bank(self):
        """
        构建小波滤波器组（虚方法，由子类实现）
        
        小波滤波器组包含：
        - real_part: 实部卷积核，形状为 [n_scales, 1, kernel_size, 1]
        - imaginary_part: 虚部卷积核（可选），形状同上
        
        Returns:
            tuple: (real_part, imaginary_part)
        """
        real_part = None
        imaginary_part = None
        return real_part, imaginary_part

    def forward(self, inputs):
        """
        计算连续小波变换（CWT），生成时频图（scalogram）
        
        如果信号有多个通道，CWT 会独立计算每个通道，最后沿通道轴堆叠。
        
        实现原理：
        1. 将输入信号与不同尺度的小波函数进行卷积
        2. 对于复小波，分别计算实部和虚部的卷积
        3. 计算复小波的幅度：|CWT| = sqrt(real^2 + imag^2)
        4. 裁剪边界以去除边界效应
        
        Args:
            inputs: (tensor) 输入信号批次
                - 形状: [batch_size, time_len] 或 [batch_size, n_channels, time_len]
                - 1D 时域信号
        
        Returns:
            scalogram: (tensor) 时频图（小波变换的幅度）
                - 形状: [batch_size, n_channels, n_scales, time_len]
                - n_scales: 频率维度（不同尺度对应不同频率）
                - time_len: 时间维度（保持或根据 stride 下采样）
        """
        # ========== 计算边界裁剪位置 ==========
        # 根据 stride 调整裁剪量（如果 stride>1，裁剪量会相应减少）
        border_crop = int(self.border_crop / self.stride)
        start = border_crop
        end = (-border_crop) if (border_crop > 0) else None
        
        # ========== 准备输入张量 ==========
        # 输入形状: [batch_size, time_len] 或 [batch_size, n_channels, time_len]
        # 扩展为 4D: [batch, 1, time_len, 1] 以适配 conv2d
        # conv2d 需要 (batch, channels, height, width) 格式
        inputs_expand = inputs.unsqueeze(1).unsqueeze(3)
        
        # ========== 实部卷积 ==========
        # 使用实部小波滤波器进行卷积
        # padding: 保持输出时间长度不变（使用 kernel_size//2 的填充）
        out_real = F.conv2d(
            input=inputs_expand, 
            weight=self.real_part,  # 形状: [n_scales, 1, kernel_size, 1]
            stride=(self.stride, 1),  # 只在时间维度使用 stride
            padding=(self.real_part.shape[2]//2, self.real_part.shape[3]//2)
        )
        # 裁剪边界并移除多余的维度
        out_real = out_real[:, :, start:end, :].squeeze(3)
        
        # ========== 虚部卷积（如果存在） ==========
        if self.imaginary_part is not None:
            # 对于复小波（如 ComplexMorlet），需要同时计算实部和虚部
            out_imag = F.conv2d(
                input=inputs_expand, 
                weight=self.imaginary_part,
                stride=(self.stride, 1), 
                padding=(self.imaginary_part.shape[2]//2, self.imaginary_part.shape[3]//2)
            )
            out_imag = out_imag[:, :, start:end, :].squeeze(3)
            
            # ========== 计算复小波的幅度 ==========
            # 复小波的幅度 = sqrt(real^2 + imag^2)
            # 这给出了信号在每个尺度和时间点的能量
            # 形状: [batch, n_channels, n_scales, time_len]
            scalogram = torch.sqrt(out_real ** 2 + out_imag ** 2)
        else:
            # 对于实小波（如 Ricker），直接使用实部结果
            scalogram = out_real
        
        return scalogram


class ComplexMorletCWT(ContinuousWaveletTransform):
    """
    复 Morlet 小波变换（Complex Morlet Wavelet Transform）
    
    复 Morlet 小波是一种常用的复值小波，由高斯包络和复指数振荡组成。
    特别适合分析瞬态信号（如引力波），因为它能同时提供良好的时间和频率分辨率。
    
    小波函数形式：
    ψ(t) = (1/√(π·σ)) · exp(-t²/σ) · exp(2πi·f₀·t)
    - 高斯包络: exp(-t²/σ) 提供时间局部化
    - 复指数: exp(2πi·f₀·t) 提供频率信息
    - σ (wavelet_width): 控制时间-频率分辨率权衡
    
    Args:
        wavelet_width: (float) 小波宽度参数（σ）
            - 控制高斯包络的宽度
            - 较小值：更好的时间分辨率，较差的频率分辨率
            - 较大值：更好的频率分辨率，较差的时间分辨率
        fs: (float) 采样频率（Hz）
            - 信号的采样率，用于将时间索引转换为实际时间
        lower_freq: (float) 最低频率（Hz）
            - scalogram 的频率范围下限
            - 对应最大的小波尺度
        upper_freq: (float) 最高频率（Hz）
            - scalogram 的频率范围上限
            - 对应最小的小波尺度
        n_scales: (int) 小波尺度数量
            - 决定频率分辨率
            - 更多尺度 = 更细的频率分辨率，但计算量更大
        size_factor: (float) 核大小因子（默认 1.0）
            - 控制小波核的大小
            - >1.0: 更大的核，更精确但更慢
            - <1.0: 更小的核，更快但可能精度降低
        trainable_width: (bool) 是否可训练小波宽度（默认 False）
            - True: 小波宽度作为可学习参数，端到端优化
            - False: 固定小波宽度
        trainable_filter: (bool) 是否可训练小波滤波器（默认 False）
            - True: 整个小波滤波器可学习
            - False: 固定小波滤波器
        border_crop: (int) 边界裁剪数量（默认 0）
            - 去除边界效应的样本数
        stride: (int) 卷积步长（默认 1）
            - 控制时间维度的下采样
    """
    def __init__(
            self,
            wavelet_width,
            fs,
            lower_freq,
            upper_freq,
            n_scales,
            size_factor=1.0,
            trainable_width=False,
            trainable_filter=False,
            border_crop=0,
            stride=1):
        # ========== 参数验证 ==========
        if lower_freq > upper_freq:
            raise ValueError("lower_freq should be lower than upper_freq")
        if lower_freq < 0:
            raise ValueError("Expected positive lower_freq.")

        # ========== 保存参数 ==========
        self.initial_wavelet_width = wavelet_width  # 初始小波宽度（用于计算核大小）
        self.fs = fs                                # 采样频率
        self.lower_freq = lower_freq                # 最低频率
        self.upper_freq = upper_freq                # 最高频率
        self.size_factor = size_factor              # 核大小因子
        self.trainable_width = trainable_width      # 是否可训练宽度
        self.trainable_filter = trainable_filter    # 是否可训练滤波器
        
        # ========== 计算小波尺度 ==========
        # 小波尺度 s 与频率 f 的关系：f = 1/s
        # 尺度越大，对应频率越低；尺度越小，对应频率越高
        s_0 = 1 / self.upper_freq  # 最小尺度（对应最高频率）
        s_n = 1 / self.lower_freq   # 最大尺度（对应最低频率）
        
        # 生成尺度数组：在 s_0 和 s_n 之间均匀分布（对数尺度）
        # 使用对数分布确保频率分辨率更均匀
        base = np.power(s_n / s_0, 1 / (n_scales - 1))  # 等比数列的公比
        self.scales = torch.from_numpy(s_0 * np.power(base, np.arange(n_scales)))
        
        # 生成对应的频率数组（用于可视化或分析）
        self.frequencies = 1 / self.scales
        
        # 调用父类初始化，构建小波滤波器组
        super().__init__(n_scales, border_crop, stride)
        

    def _build_wavelet_bank(self):
        """
        构建复 Morlet 小波滤波器组
        
        为每个尺度生成一个小波核，所有核组成滤波器组。
        小波核包含实部和虚部，用于计算复小波变换。
        """
        # ========== 创建可训练的小波宽度参数 ==========
        self.wavelet_width = nn.Parameter(
            data=torch.tensor(self.initial_wavelet_width, dtype=torch.float32),
            requires_grad=self.trainable_width  # 如果 trainable_width=True，可以端到端优化
        )
        
        # ========== 计算小波核的大小 ==========
        # 为了确保小波在截断后仍有足够精度，需要计算合适的核大小
        # 截断大小基于最大尺度和小波宽度
        # 公式：|t| < truncation_size，确保小波能量集中在核内
        truncation_size = self.scales.max() * np.sqrt(4.5 * self.initial_wavelet_width) * self.fs
        one_side = int(self.size_factor * truncation_size)  # 核的半宽
        kernel_size = 2 * one_side + 1  # 核的总大小（奇数，确保有中心点）
        
        # ========== 生成时间数组 ==========
        # k_array: 采样点索引 [-one_side, ..., 0, ..., one_side]
        k_array = np.arange(kernel_size, dtype=np.float32) - one_side
        t_array = k_array / self.fs  # 转换为实际时间（秒）
        
        # ========== 为每个尺度生成小波核 ==========
        wavelet_bank_real = []  # 实部核列表
        wavelet_bank_imag = []  # 虚部核列表
        
        for scale in self.scales:
            # 归一化常数：确保小波能量归一化
            # 公式：1/√(π·σ) · scale · fs / 2
            norm_constant = torch.sqrt(np.pi * self.wavelet_width) * scale * self.fs / 2.0
            
            # 缩放时间：t' = t / scale
            # 不同尺度对应不同的时间尺度
            scaled_t = t_array / scale
            
            # 高斯包络：exp(-t'²/σ)
            # 提供时间局部化，控制小波的时间宽度
            exp_term = torch.exp(-(scaled_t ** 2) / self.wavelet_width)
            
            # 小波基函数：高斯包络 / 归一化常数
            kernel_base = exp_term / norm_constant
            
            # 复指数部分：exp(2πi·t') = cos(2π·t') + i·sin(2π·t')
            # 实部：cos(2π·t')
            kernel_real = kernel_base * np.cos(2 * np.pi * scaled_t)
            # 虚部：sin(2π·t')
            kernel_imag = kernel_base * np.sin(2 * np.pi * scaled_t)
            
            wavelet_bank_real.append(kernel_real)
            wavelet_bank_imag.append(kernel_imag)
        
        # ========== 堆叠所有尺度的小波核 ==========
        # 形状: [n_scales, kernel_size]
        wavelet_bank_real = torch.stack(wavelet_bank_real, axis=0)
        wavelet_bank_imag = torch.stack(wavelet_bank_imag, axis=0)
        
        # ========== 调整为卷积所需的形状 ==========
        # conv2d 需要的权重形状: [out_channels, in_channels, height, width]
        # 这里: [n_scales, 1, kernel_size, 1]
        # - n_scales: 输出通道数（每个尺度一个通道）
        # - 1: 输入通道数（单通道信号）
        # - kernel_size: 卷积核高度（时间维度）
        # - 1: 卷积核宽度（固定为1，因为是1D信号）
        wavelet_bank_real = wavelet_bank_real.unsqueeze(1).unsqueeze(3)
        wavelet_bank_imag = wavelet_bank_imag.unsqueeze(1).unsqueeze(3)
        
        # ========== 注册为模型参数 ==========
        # 如果 trainable_filter=True，这些参数可以在训练中更新 默认是False
        self.real_part = nn.Parameter(
            data=wavelet_bank_real,
            requires_grad=self.trainable_filter
        )
        self.imaginary_part = nn.Parameter(
            data=wavelet_bank_imag,
            requires_grad=self.trainable_filter
        )