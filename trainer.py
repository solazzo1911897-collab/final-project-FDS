from pathlib import Path
from tqdm import tqdm
from copy import copy, deepcopy
from collections import defaultdict
import time
import pickle
import subprocess
import inspect
import sys
import os
import uuid
from pprint import pformat
import __main__
import resource

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn import SyncBatchNorm

from .utils import get_device, seed_everything, get_gpu_memory
from .tb_logger import DummyTensorBoardLogger
from .callbacks import (
    TorchLogger, DummyLogger, SaveSnapshot
)
from .hooks import TrainHook
from . import distributed as comm
from .clip_grad import dispatch_clip_grad

try:
    from torch.cuda import amp
    AMP = True
except ModuleNotFoundError:
    AMP = False

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.serialization as xser
    import torch_xla.distributed.xla_multiprocessing as xmp
    XLA = True
except ModuleNotFoundError:
    XLA = False


class TorchTrainer:
    '''
    Simple Trainer for PyTorch models
    
    This is something similar to PyTorch Lightning, but this works with vanilla PyTorch modules.
    
    中文：一个轻量级的 PyTorch 训练器，核心思路类似于
    PyTorch Lightning，但保持对原生 PyTorch 模块的最小侵入。
    通过统一的注册/训练流程，简化单卡、DP、DDP、XLA 等多场景的训练管理。
    '''

    def __init__(self,
                 model, device=None, serial='trainer0'):
        # 中文：serial 作为当前训练实例的唯一标识，用于日志和快照文件名

        self.serial = serial
        self.device, self.device_ids = get_device(device)
        self.xla = self.device.type == 'xla'
        self.world_size = len(self.device_ids)
        self.model = model
        self.rank = 0
        self._register_ready = False
        self._model_ready = False

        ### Implicit attributes
        # DDP
        self.ddp_sync_batch_norm = True
        self.ddp_average_loss = True
        self.ddp_sync_last = False # deprecated
        self.ddp_workers = -1
        # MISC
        self.debug = False

    def _register_callbacks(self, callbacks):
        '''
        注册回调函数（Callbacks）
        
        将 callbacks 列表中的每个回调对象的方法按类型分类存储：
        - before_epoch: 每个 epoch 开始前执行的函数列表
        - after_epoch: 每个 epoch 结束后执行的函数列表
        - _save_snapshot: 保存模型快照时执行的函数列表
        - _load_snapshot: 加载模型快照时执行的函数列表
        
        这些回调函数会在训练循环的相应时机被调用，实现日志记录、模型保存等功能。
        '''
        self.before_epoch = [func.before_epoch for func in callbacks]
        self.after_epoch = [func.after_epoch for func in callbacks]
        self._save_snapshot = [func.save_snapshot for func in callbacks]
        self._load_snapshot = [func.load_snapshot for func in callbacks]

    def _register_hook(self, hook):
        '''
        注册训练钩子（Hook）
        
        将 TrainHook 对象的方法注册为训练器的成员方法，使得训练循环可以直接调用：
        - forward_train: 训练时的前向传播，返回 (loss, predictions)
        - forward_valid: 验证时的前向传播，返回 (loss, predictions)
        - forward_test: 测试/推理时的前向传播，返回 predictions
        - evaluate_batch: 每个 batch 的评估，用于计算 batch 级别的指标
        - evaluate_epoch: 每个 epoch 的评估，用于计算 epoch 级别的指标
        
        这样设计使得训练逻辑与具体的前向传播实现解耦，用户可以通过自定义 Hook 来定义自己的训练流程。
        '''
        self.forward_train = hook.forward_train
        self.forward_valid = hook.forward_valid
        self.forward_test = hook.forward_test
        self.evaluate_batch = hook.evaluate_batch
        self.evaluate_epoch = hook.evaluate_epoch

    def _configure_model(self):
        ''' Mixed precision '''
        # 中文：根据 fp16/amp 开关选择混合精度策略，若环境不支持则自动回退
        if self.fp16:
            if AMP:
                if self.rank == 0:
                    self.logger('Mixed precision training on torch amp.')
            else:
                self.fp16 = False
                if self.rank == 0:
                    self.logger('No mixed precision training backend found.')

        ''' Parallel training '''
        # 中文：根据 parallel 配置将模型放置到对应设备并包装为 DP/DDP/XLA。
        # - self.xla=True：走 XLA 分布式，模型直接放到 XLA device（TPU/云 TPU 等）。 XLA（Accelerated Linear Algebra）是 Google 的编译与执行框架，主要用于加速深度学习计算，常见场景是 TPU/云 TPU 或支持 XLA 的设备。PyTorch 通过 torch_xla 提供 XLA 后端，允许模型在 TPU 上运行，使用类 TPU 的分布式训练（与 CUDA 上的 DDP 不同）。在代码中 self.xla=True 就表示当前设备是 XLA 设备，训练流程会走 XLA 的数据加载、分布式与优化器步进逻辑。
        # - parallel='dp'：单进程多卡 DataParallel，自动切分 batch。
        # - parallel='ddp'：多进程多卡 DistributedDataParallel，需 per-rank 设备绑定。
        # - None：单卡/CPU 直连。
        if self.xla:  # DDP on xla
            self.model.to(self.device)
            if self.rank == 0:
                self.logger(f'Model on {self.device}')

        elif self.parallel == 'dp': # DP on cuda
            # 中文：DataParallel 使用主进程控制多卡，梯度在主卡归并
            self.model = DataParallel(
                self.model, device_ids=self.device_ids).to(self.device)
            if hasattr(self, 'criterion'):
                self.criterion = self.criterion.to(self.device)
            self.logger(f'DataParallel on devices {self.device_ids}')

        elif self.parallel == 'ddp': # DDP on cuda
            # 中文：DDP 需确保每个进程只用一张卡；可选 SyncBatchNorm 做跨卡同步
            if self.ddp_sync_batch_norm:
                self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DistributedDataParallel(
                self.model.to(self.rank), device_ids=[self.rank],
                broadcast_buffers=False,
                find_unused_parameters=True
            )
            if hasattr(self, 'criterion'):
                self.criterion = self.criterion.to(self.rank)
            if self.rank == 0:
                self.logger(
                    f'DistributedDataParallel on devices {self.device_ids}')

        elif self.parallel is not None:
            raise ValueError(f'Unknown type of parallel {self.parallel}')

        else:  # Single device
            # 中文：单设备/CPU 情况，直接将模型与损失函数放入目标设备
            self.model.to(self.device)
            if hasattr(self, 'criterion'):
                self.criterion = self.criterion.to(self.device)
            self.logger(f'Model on {self.device}')
        
        self._model_ready = True

    def _configure_loader_ddp(self, loader, shuffle=True):
        if loader is None:
            return None
        # 中文：重建 DataLoader，使用 DistributedSampler 保证各进程数据切分一致
        skip_keys = ['sampler', 'batch_sampler', 'dataset_kind']
        dl_args = {
            k: v for k, v in loader.__dict__.items()
            if not k.startswith('_') and k not in skip_keys
        }
        sampler = DistributedSampler(
            loader.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)
        dl_args['sampler'] = sampler
        if self.ddp_workers == -1:
            dl_args['num_workers'] = int(
                dl_args['num_workers'] / self.world_size)
        else:
            dl_args['num_workers'] = self.ddp_workers
        if dl_args['batch_size'] % self.world_size != 0:
            raise ValueError(
                f'batch size must be a multiple of world size({self.world_size}).')
        dl_args['batch_size'] = int(dl_args['batch_size'] / self.world_size)
        if self.xla:
            return pl.ParallelLoader(type(loader)(**dl_args), [self.device])
        else:
            return type(loader)(**dl_args)

    def _train_one_epoch(self, loader):
        """
        训练一个完整的 epoch
        
        这是训练循环的核心函数，负责：
        1. 遍历训练数据加载器中的所有批次
        2. 执行前向传播、计算损失、反向传播和参数更新
        3. 支持多种训练模式：FP16混合精度、SAM优化器、DDP分布式训练、XLA设备
        4. 收集和统计训练指标（损失、指标、学习率等）
        5. 检测NaN值并记录警告
        6. 统计数据加载和训练耗时
        
        Args:
            loader: 训练数据加载器（DataLoader），提供批次数据
            
        Returns:
            tuple: (loss_total, metric_total, monitor_metrics_total)
                - loss_total: 整个epoch的平均损失值
                - metric_total: 整个epoch的平均评估指标值
                - monitor_metrics_total: 监控指标列表
        """
        # ========== 初始化计时器和存储结构 ==========
        loader_time = .0  # 数据加载累计耗时
        train_time = .0   # 训练计算累计耗时
        start_time = time.time()  # epoch开始时间
        curr_time = time.time()   # 当前时间戳

        # 初始化epoch级别的存储字典，用于收集整个epoch的指标
        # - approx: 模型预测值
        # - target: 真实标签
        # - loss: 每个batch的损失值
        # - batch_metric: 每个batch的评估指标
        self.epoch_storage = defaultdict(list)
        for key in ['approx', 'target', 'loss', 'batch_metric']:
            self.epoch_storage[key] = []
        
        # 如果使用FP16混合精度训练，创建梯度缩放器
        # GradScaler用于防止FP16训练中的梯度下溢问题
        if self.fp16:
            scaler = amp.GradScaler()

        # 将模型设置为训练模式（启用dropout、batch normalization的更新等）
        self.model.train()
        
        # 根据是否启用进度条和是否为主进程，选择迭代器类型
        # 只有主进程（rank=0）才显示进度条，避免多进程时重复输出
        if self.progress_bar and self.rank == 0:
            iterator = enumerate(tqdm(loader, desc='train'))
        else:
            iterator = enumerate(loader)
        batch_total = len(loader)  # 总批次数
        ett_disp = False  # 是否已显示预计训练时间（Estimated Training Time）

        # ========== 遍历所有批次进行训练 ==========
        for batch_i, inputs in iterator:
            # 统计数据加载耗时（从上一个batch结束到当前batch数据加载完成）
            loader_time += time.time() - curr_time
            curr_time = time.time()
            elapsed_time = curr_time - start_time
            
            # ========== 首个epoch训练30秒后显示预计耗时和资源占用 ==========
            # 这有助于用户了解训练速度和资源消耗情况
            if self.state['epoch'] == 0 and elapsed_time > 30 and not ett_disp: # show ETA
                # 根据已训练时间估算整个epoch的预计耗时
                ett = elapsed_time * batch_total // batch_i
                self.logger(f'Estimated epoch training time: {int(ett)} s')
                # 尝试获取最大内存使用量（RAM）
                try:
                    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
                    self.logger(f'Maximum RAM usage: {ram_usage} MB')
                except:
                    self.logger('Failed to get RAM usage.')
                # 尝试获取最大GPU显存使用量（GRAM）
                try:
                    gram_usage = int(max(get_gpu_memory().values()))
                    self.logger(f'Maximum GRAM usage: {gram_usage} MB')
                except:
                    self.logger('Failed to get GRAM usage.')
                ett_disp = True  # 标记已显示，避免重复输出

            # ========== 准备训练 ==========
            # 清零优化器的梯度（防止梯度累积）
            self.optimizer.zero_grad()
            # 计算已完成的批次总数（用于学习率调度和日志记录）
            batches_done = batch_total * (self.global_epoch-1) + batch_i
            # 将输入数据移动到指定设备（CPU/GPU/TPU）
            inputs = [t.to(self.device) for t in inputs]
            
            # ========== 前向传播和反向传播 ==========
            # 根据是否使用FP16混合精度和SAM优化器，选择不同的训练策略
            # 主要分为四种情况：
            # 1. FP16 + SAM: 使用混合精度和SAM的两步优化
            # 2. FP16 + 普通优化器: 使用混合精度和标准优化
            # 3. FP32 + XLA + SAM: XLA设备不支持SAM，会报错
            # 4. FP32 + SAM: 使用FP32精度和SAM的两步优化
            # 5. FP32 + 普通优化器: 标准训练流程
            
            if self.fp16:
                # ========== FP16混合精度训练 ==========
                # 在autocast上下文中执行前向传播（自动转换为FP16）
                with amp.autocast():
                    # 前向传播：计算损失和模型输出
                    loss, approx = self.forward_train(self, inputs)
                    # 评估当前batch的指标（如准确率、AUC等）
                    self.evaluate_batch(self, inputs, approx) # evaluation
                
                # 梯度累积：将损失除以累积步数，使得多次累积后的总梯度等于单次训练的梯度
                loss = loss / self.grad_accumulations
                # 使用梯度缩放器进行反向传播（防止FP16梯度下溢）
                scaler.scale(loss).backward()
                
                # 如果启用了梯度裁剪，对梯度进行裁剪（防止梯度爆炸）
                if self.clip_grad is not None:
                    dispatch_clip_grad(self.model.parameters(), self.max_grad_norm, mode=self.clip_grad)
                
                # 当达到梯度累积的步数时，执行参数更新
                if (batch_i + 1) % self.grad_accumulations == 0:
                    if self.sam:
                        # ========== FP16 + SAM优化器（Sharpness-Aware Minimization）==========
                        # SAM是一种两阶段优化器，通过最小化损失函数的尖锐度来提高泛化能力
                        
                        # 第一步：在当前位置计算梯度并执行第一次优化步骤
                        optimizer_state = scaler._per_optimizer_states[id(self.optimizer)]
                        scaler.unscale_(self.optimizer)  # 取消梯度缩放，准备检查NaN
                        # 检查是否有NaN/Inf梯度，如果没有则执行第一步
                        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
                            self.optimizer.first_step(zero_grad=True)  # SAM的第一步：移动到尖锐区域
                        optimizer_state["stage"] = 2
                        scaler.update()  # 更新缩放因子
                        
                        # 第二步：在第一步后的位置重新计算梯度并执行第二次优化步骤
                        with amp.autocast():
                            loss2, _ = self.forward_train(self, inputs)  # 在扰动后的位置计算损失
                        scaler.scale(loss2).backward()  # 反向传播
                        scaler.unscale_(self.optimizer)
                        # 再次检查NaN/Inf，如果没有则执行第二步
                        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
                            self.optimizer.second_step(zero_grad=True)  # SAM的第二步：回到平坦区域
                        optimizer_state["stage"] = 2
                        scaler.update()
                    else:
                        # ========== FP16 + 普通优化器 ==========
                        # 标准优化步骤：更新参数并更新梯度缩放器
                        scaler.step(self.optimizer)  # 执行优化步骤
                        scaler.update()  # 更新缩放因子（根据是否有NaN/Inf调整）
            else:
                # ========== FP32标准精度训练 ==========
                # 前向传播：计算损失和模型输出
                loss, approx = self.forward_train(self, inputs)
                # 评估当前batch的指标
                self.evaluate_batch(self, inputs, approx) # evaluation
                # 梯度累积：将损失除以累积步数
                loss = loss / self.grad_accumulations
                # 反向传播：计算梯度
                loss.backward()
                
                # 如果启用了梯度裁剪，对梯度进行裁剪
                if self.clip_grad is not None:
                    dispatch_clip_grad(self.model.parameters(), self.max_grad_norm, mode=self.clip_grad)
                
                # 当达到梯度累积的步数时，执行参数更新
                if (batch_i + 1) % self.grad_accumulations == 0:
                    if self.xla:
                        # ========== XLA设备（TPU）训练 ==========
                        # XLA设备不支持SAM优化器
                        if self.sam:
                            raise RuntimeError('SAM optimizer on XLA device is not available.')
                        else:
                            # XLA设备使用特殊的优化步骤，barrier=True确保同步
                            xm.optimizer_step(self.optimizer, barrier=True)
                    else:
                        # ========== CUDA/CPU设备训练 ==========
                        if self.sam:
                            # FP32 + SAM优化器：两阶段优化
                            self.optimizer.first_step(zero_grad=True)   # 第一步：移动到尖锐区域
                            loss2, _ = self.forward_train(self, inputs)  # 在扰动位置重新计算损失
                            loss2.backward()  # 反向传播
                            self.optimizer.second_step(zero_grad=True)  # 第二步：回到平坦区域
                        else:
                            # FP32 + 普通优化器：标准优化步骤
                            self.optimizer.step()
                    
                    # 如果使用batch级别的学习率调度器，在每个batch后更新学习率
                    if self.batch_scheduler:
                        self.scheduler.step()
            
            # ========== NaN检测和警告 ==========
            # 检测损失值中是否包含NaN（通常表示训练不稳定或数值溢出）
            if torch.isnan(loss).any():
                self.logger(f'{torch.isnan(loss).sum()} NaN detected in loss. ({batch_i}/{len(loader)})')
            # 检测模型输出中是否包含NaN（可能影响后续评估）
            if torch.isnan(approx).any():
                self.logger(f'{torch.isnan(approx).sum()} NaN detected in output tensor. ({batch_i}/{len(loader)})')
            
            # ========== 处理分布式训练中的损失值 ==========
            # 在DDP模式下，如果启用了损失平均，需要从所有进程收集损失并求平均
            if self.parallel == 'ddp' and self.ddp_average_loss:
                if self.xla:
                    # XLA设备使用all_gather收集所有进程的损失
                    loss_batch = xm.all_gather(
                        loss.detach().clone().view(1)).mean().item()
                else:
                    # CUDA设备使用自定义的gather_tensor函数收集损失
                    loss_batch = comm.gather_tensor(
                        loss.detach().clone().view(1)).mean().item()
            else:
                # 单进程训练或使用主进程的损失值
                loss_batch = loss.item()

            # ========== 记录batch级别的日志 ==========
            # 获取当前学习率（支持多参数组的情况）
            learning_rate = [param_group['lr']
                             for param_group in self.optimizer.param_groups]
            # 准备TensorBoard日志
            logs = [
                ('batch_train_loss', loss_batch),
                ('batch_train_lr', learning_rate)
            ]
            # 如果有batch级别的评估指标，也记录到日志
            if len(self.epoch_storage['batch_metric']) > 0:
                metric = self.epoch_storage['batch_metric'][-1]
                logs.append(('batch_valid_mertric', metric))
            # 写入TensorBoard日志
            self.tb_logger.list_of_scalars_summary(logs, batches_done)
            # 保存损失值到epoch存储中
            self.epoch_storage['loss'].append(loss_batch)

            # 统计训练计算耗时（从数据加载完成到当前batch训练完成）
            train_time += time.time() - curr_time
            curr_time = time.time()

        # ========== 调试模式：输出耗时统计 ==========
        # 仅在调试模式且为主进程时输出，避免多进程重复日志
        if self.debug and self.rank == 0:
            self.logger(
                f'loader: {loader_time:.1f} s | train: {train_time:.1f} s')

        # ========== 整理epoch存储的数据 ==========
        # 将收集到的列表数据转换为张量，便于后续计算
        for key, val in self.epoch_storage.items():
            if len(val) > 0:
                if isinstance(val[0], torch.Tensor):
                    # 如果元素是张量，使用cat拼接
                    self.epoch_storage[key] = torch.cat(val)
                else:
                    # 如果元素是标量，转换为张量并移动到设备
                    self.epoch_storage[key] = torch.tensor(val).to(self.device)

        # 计算整个epoch的平均损失
        loss_total = self.epoch_storage['loss'].mean().item()

        # ========== DDP模式：汇总所有进程的数据 ==========
        # 在分布式训练中，需要从所有进程收集数据，确保评估基于全局数据
        if self.parallel == 'ddp':
            # 遍历所有存储的键，收集各进程的张量/指标
            for key, val in self.epoch_storage.items():
                if len(val) > 0:
                    if self.xla:
                        # XLA设备使用all_gather收集所有进程的数据
                        self.epoch_storage[key] = xm.all_gather(val)
                    else:
                        # CUDA设备使用自定义的gather_tensor函数收集数据
                        self.epoch_storage[key] = comm.gather_tensor(val)

            # 基于汇总后的全局数据评估整个epoch的指标
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        else:
            # ========== 单进程模式：直接评估 ==========
            # 单进程训练，直接使用当前进程的数据进行评估
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        # ========== 处理评估指标 ==========
        # 如果没有评估指标（eval_metric未设置），使用损失值作为指标
        if metric_total is None:
            metric_total = loss_total

        # ========== 记录epoch级别的日志 ==========
        # 准备TensorBoard日志
        logs = [
            ('epoch_train_loss', loss_total),
            ('epoch_train_metric', metric_total),
        ]
        # 写入TensorBoard日志
        self.tb_logger.list_of_scalars_summary(logs, self.global_epoch)
        
        # 返回epoch级别的指标
        return loss_total, metric_total, monitor_metrics_total

    def _valid_one_epoch(self, loader):
        # 中文：验证循环，不做反向传播；同样收集 batch/epoch 指标
        self.epoch_storage = defaultdict(list)
        for key in ['approx', 'target', 'loss', 'batch_metric']:
            self.epoch_storage[key] = []

        self.model.eval()
        if self.progress_bar and self.rank == 0:
            iterator = enumerate(tqdm(loader, desc='valid'))
        else:
            iterator = enumerate(loader)

        with torch.no_grad():
            for batch_i, inputs in iterator:
                batches_done = len(loader) * (self.global_epoch-1) + batch_i
                inputs = [t.to(self.device) for t in inputs]
                loss, approx = self.forward_valid(self, inputs)
                if torch.isnan(loss).any():
                    self.logger(f'{torch.isnan(loss).sum()} NaN detected in loss. ({batch_i}/{len(loader)})')
                if torch.isnan(approx).any():
                    self.logger(f'{torch.isnan(approx).sum()} NaN detected in output tensor. ({batch_i}/{len(loader)})')
                self.evaluate_batch(self, inputs, approx)
                if self.parallel == 'ddp' and self.ddp_average_loss:
                    if self.xla:
                        loss_batch = xm.all_gather(
                            loss.detach().clone().view(1)).mean().item()
                    else:
                        loss_batch = comm.gather_tensor(
                            loss.detach().clone().view(1)).mean().item()
                else:  # Use loss on device: 0
                    loss_batch = loss.item()

                logs = [
                    ('batch_valid_loss', loss_batch),
                ]
                if len(self.epoch_storage['batch_metric']) > 0:
                    metric = self.epoch_storage['batch_metric'][-1]
                    logs.append(('batch_valid_mertric', metric))
                self.tb_logger.list_of_scalars_summary(logs, batches_done)
                self.epoch_storage['loss'].append(loss_batch)

        for key, val in self.epoch_storage.items():
            if len(val) > 0:
                if isinstance(val[0], torch.Tensor):
                    self.epoch_storage[key] = torch.cat(val)
                else:
                    self.epoch_storage[key] = torch.tensor(val).to(self.device)

        loss_total = self.epoch_storage['loss'].mean().item()

        if self.parallel == 'ddp':
            for key, val in self.epoch_storage.items():
                if len(val) > 0:
                    if self.xla:
                        self.epoch_storage[key] = xm.all_gather(val)
                    else:
                        self.epoch_storage[key] = comm.gather_tensor(val)

            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        else:
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        if metric_total is None:
            metric_total = loss_total

        logs = [
            ('epoch_valid_loss', loss_total),
            ('epoch_valid_metric', metric_total),
        ]
        self.tb_logger.list_of_scalars_summary(logs, self.global_epoch)
        return loss_total, metric_total, monitor_metrics_total

    def _train(self, loader, loader_valid, num_epochs):
        '''
        训练循环的核心驱动函数
        
        这是训练器的主要训练逻辑，负责协调整个训练过程。每个 epoch 的执行流程如下：
        1. before_epoch callbacks（epoch 开始前的回调）
        2. 训练循环（_train_one_epoch）
        3. 验证循环（_valid_one_epoch，如果提供了验证集）
        4. 更新训练状态（loss、metric、learning_rate 等）
        5. 更新学习率调度器（如果不是 batch scheduler）
        6. after_epoch callbacks（epoch 结束后的回调，包括日志记录）
        7. 保存状态到历史记录
        8. 检查是否需要保存模型或提前停止
        
        参数:
            loader: 训练数据加载器
            loader_valid: 验证数据加载器，如果为 None 则跳过验证
            num_epochs: 训练的 epoch 数量
        
        注意:
            - 必须在调用前确保 _register_ready 和 _model_ready 为 True
            - DDP 模式下，每个 epoch 需要设置 sampler 的 epoch 以确保数据打乱的一致性
            - 非主进程在非调试模式下会跳过 logger.after_epoch 以避免重复日志
            - 如果设置了 early stop，会在满足条件时提前终止训练
        '''
        assert self._register_ready
        assert self._model_ready
        # 中文：标准训练驱动，按 epoch 依次执行 callback -> train -> valid -> callback
        for epoch in range(num_epochs):
            # 中文：DDP 模式下，每个 epoch 需要设置 sampler 的 epoch，确保每个进程的数据打乱一致
            if self.parallel == 'ddp' and not self.xla:
                loader.sampler.set_epoch(epoch)
            
            # 中文：更新当前 epoch 编号到状态中
            self.state.update({'epoch': epoch})

            # 中文：执行 epoch 开始前的回调函数（如重置某些状态、打印信息等）
            ''' before epoch callbacks '''
            for func in self.before_epoch:
                func(self)

            # 中文：执行一个完整的训练 epoch，返回训练损失、指标和监控指标
            ''' Training loop '''
            loss_train, metric_train, monitor_metrics_train = \
                self._train_one_epoch(loader)

            # 中文：执行验证循环（如果提供了验证集）
            # 返回验证损失、指标和监控指标；如果没有验证集，则返回 None
            ''' Validation loop '''
            if loader_valid is None:
                loss_valid, metric_valid, monitor_metrics_valid = \
                    None, None, None
            else:
                # 中文：DDP 模式下，验证集的 sampler 也需要设置 epoch
                if self.parallel == 'ddp' and not self.xla:
                    loader_valid.sampler.set_epoch(epoch)
                loss_valid, metric_valid, monitor_metrics_valid = \
                    self._valid_one_epoch(loader_valid)

            # 中文：将本 epoch 的训练和验证结果更新到状态字典中
            # 包括损失、指标、监控指标和学习率等信息
            self.state.update({
                'epoch': epoch,
                'train_loss': loss_train,
                'train_metric': metric_train,
                'train_monitor': monitor_metrics_train,
                'valid_loss': loss_valid,
                'valid_metric': metric_valid,
                'valid_monitor': monitor_metrics_valid,
                'learning_rate': [group['lr'] for group in self.optimizer.param_groups][0]
            })

            # 中文：如果不是 batch scheduler，则在每个 epoch 结束后更新学习率
            # 如果指定了 scheduler_target（如 'valid_loss'），则根据该指标调整学习率
            if not self.batch_scheduler:  # Epoch scheduler
                if self.scheduler_target is not None:
                    self.scheduler.step(self.state[self.scheduler_target])
                else:
                    self.scheduler.step()

            # 中文：执行 epoch 结束后的回调函数
            # 包括用户注册的回调（如 EarlyStopping、SaveSnapshot）和日志记录回调
            # DDP 模式下，非主进程且非调试模式时跳过 logger.after_epoch 避免重复日志
            ''' After epoch callbacks '''
            after_trains = self.after_epoch + [self.logger.after_epoch] # 找到了
            if self.rank != 0 and not self.debug:
                after_trains = after_trains[:-1]  # 移除 logger.after_epoch
            for func in after_trains:
                func(self)
            
            # 中文：将当前 epoch 的状态保存到历史记录中，用于后续分析和导出 DataFrame
            self._states.append(self.state.copy())

            # 中文：如果设置了 checkpoint 标志（通常由回调函数设置），则保存模型快照
            # 只在主进程执行，避免多进程重复保存
            if self.checkpoint and self.rank == 0:
                ''' Save model '''
                self.save_snapshot()
                self.checkpoint = False

            # 中文：检查是否触发提前停止（通常由 EarlyStopping 回调设置）
            # 如果 stop_train 为 True，则终止训练循环
            if self.stop_train:
                ''' Early stop '''
                if self.rank == 0:
                    self.logger('Training stopped by overfit detector.')
                break

            # 中文：更新全局 epoch 计数器
            self.global_epoch += 1

        # 中文：DDP 训练结束后，销毁进程组，释放资源
        if self.parallel == 'ddp':
            dist.destroy_process_group()

    def _train_ddp(self, rank, dist_url, loader, loader_valid, num_epochs):
        # 中文：DDP 子进程入口，设置随机种子、进程组并绑定设备
        seed_everything(self.random_state, self.deterministic)
        self.rank = rank
        dist.init_process_group(
            backend='nccl', init_method=dist_url,
            world_size=self.world_size, rank=rank)
        comm.sync()
        torch.cuda.set_device(self.rank)
        if self.rank == 0:
            self.logger(f'All processes initialized.')

        ''' Configure model and loader '''
        self._configure_model()
        loader = self._configure_loader_ddp(loader)
        loader_valid = self._configure_loader_ddp(loader_valid, shuffle=False)

        ''' Train '''
        self._train(loader, loader_valid, num_epochs)

    def _train_xla(self, rank, loader, loader_valid, num_epochs):
        # 中文：XLA 设备训练入口，使用 XLA loader 并分发到 TPU core
        seed_everything(self.random_state, self.deterministic)
        self.device = xm.xla_device()
        self.rank = xm.get_ordinal()
        self._configure_model()
        loader = self._configure_loader_ddp(loader)
        loader_valid = self._configure_loader_ddp(loader_valid, shuffle=False)
        self._train(
            loader.per_device_loader(self.device),
            loader_valid.per_device_loader(self.device),
            num_epochs)

    def predict(self, loader, parallel=None, progress_bar=False):
        # 中文：仅前向推理流程，不依赖梯度；DDP 推理未实现
        self.parallel = parallel
        if not self._register_ready: # is hook and callbacks registered?
            raise AttributeError('Register hook and callbacks by .register() method.')
        if not self._model_ready: # is model configured?
            self.fp16 = False
            self.logger = DummyLogger('')
            if parallel == 'ddp':
                raise NotImplementedError('DDP prediction is not implemented.')
            else:
                self._configure_model()
                
        if progress_bar:
            iterator = tqdm(loader, desc='inference')
        else:
            iterator = loader
        prediction = []
        self.model.eval()
        with torch.no_grad():
            for inputs in iterator:
                inputs = [t.to(self.device) for t in inputs]
                approx = self.forward_test(self, inputs)
                prediction.append(approx.detach())
        prediction = torch.cat(prediction).cpu().numpy()

        return prediction

    def save_snapshot(self, path=None):
        # 中文：遍历回调执行保存逻辑；默认回调会把模型/优化器等状态写入文件
        for func in self._save_snapshot:
            func(self, path)

    def load_snapshot(self, path=None, device=None):
        # 中文：与保存对应，从快照恢复；可选择加载到指定 device
        for func in self._load_snapshot:
            func(self, path, device)

    def register(self, hook=TrainHook(), callbacks=[SaveSnapshot()]):
        '''
        注册训练所需的 Hook 和 Callbacks
        
        这是训练器初始化的关键步骤，必须在训练前调用。
        
        参数:
            hook: TrainHook 实例，提供前向传播和评估的接口
                - forward_train: 训练时的前向传播逻辑
                - forward_valid: 验证时的前向传播逻辑
                - forward_test: 测试/推理时的前向传播逻辑
                - evaluate_batch: 每个 batch 的评估逻辑
                - evaluate_epoch: 每个 epoch 的评估逻辑
            
            callbacks: 回调函数列表，用于在训练过程中执行特定操作
                - before_epoch: epoch 开始前的回调
                - after_epoch: epoch 结束后的回调
                - save_snapshot: 保存模型快照的回调
                - load_snapshot: 加载模型快照的回调
        
        作用:
            1. 将 hook 的方法注册为训练器的成员方法，便于在训练循环中调用
            2. 将 callbacks 的方法按类型分类存储，在相应时机触发
            3. 设置 _register_ready 标志，表示训练器已准备好开始训练
        '''
        self._register_hook(hook)
        self._register_callbacks(callbacks)
        self._register_ready = True

    def train(self,
              # Essential
              criterion, optimizer, scheduler, loader, num_epochs,
              batch_scheduler=False, scheduler_target=None,
              hook=TrainHook(), callbacks=[SaveSnapshot()],
              # Evaluation
              loader_valid=None, eval_metric=None, monitor_metrics=[],
              # Snapshot
              export_dir=None, resume=False,
              # Training option
              fp16=False, parallel=None, grad_accumulations=1, 
              deterministic=None, random_state=0, 
              clip_grad=None, max_grad_norm=10000, 
              # Logging
              logger=None, tb_logger=None, progress_bar=False, 
              ):
        # Register params
        # 中文：接收训练所需的核心对象与开关配置，存入实例属性
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_scheduler = batch_scheduler
        self.scheduler_target = scheduler_target
        self.grad_accumulations = grad_accumulations
        self.deterministic = deterministic
        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.monitor_metrics = monitor_metrics
        self.logger = logger
        self.tb_logger = tb_logger
        self.fp16 = fp16
        self.parallel = parallel
        self.progress_bar = progress_bar
        # 中文：注册 Hook 和 Callbacks，这是训练器初始化的关键步骤
        # - hook 定义了前向传播和评估的具体实现（forward_train/valid/test, evaluate_batch/epoch）
        # - callbacks 定义了训练过程中的回调逻辑（before_epoch, after_epoch, save/load_snapshot）
        # 注册后，训练器才能正确执行训练循环中的各个步骤
        self.register(hook=hook, callbacks=callbacks)

        # Important flags
        # 中文：训练状态标记，控制 early stop、保存等逻辑
        self.global_epoch = 1
        self.stop_train = False
        self.checkpoint = False
        self.outoffold = None
        self.prediction = None
        if self.optimizer.__class__.__name__ == 'SAM':
            self.sam = True
        else:
            self.sam = False

        ''' Configure directory '''
        if export_dir is None:
            export_dir = Path().cwd()
        elif isinstance(export_dir, str):
            export_dir = Path(export_dir).expanduser()
        assert len(export_dir.suffix) == 0  # export_dir must be directory
        export_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = export_dir
        self.snapshot_path = self.base_dir / f'{self.serial}.pt' 

        ''' Configure loggers '''
        if self.logger is None:
            self.logger = TorchLogger(self.base_dir / f'{self.serial}.log')
        elif isinstance(self.logger, (str, Path)):
            self.logger = TorchLogger(self.logger, file=True)
        elif isinstance(self.logger, TorchLogger):
            pass
        else:
            raise ValueError('Invalid type of logger.')

        if self.tb_logger is None:
            self.tb_logger = DummyTensorBoardLogger()

        ''' Configure loss function and metrics '''
        # 中文：若未指定 eval_metric，则默认用 criterion 作为度量；保证 monitor_metrics 为列表
        if eval_metric is None:
            self.logger(
                'eval_metric is not set. criterion will be used.')
        if not isinstance(self.monitor_metrics, (list, tuple)):
            self.monitor_metrics = [self.monitor_metrics]

        ''' Resume training '''
        # 中文：resume=True 时从快照中恢复 state 与 epoch 计数
        if resume:
            self.load_snapshot(self.snapshot_path, device='cpu')
            self.global_epoch += 1
            self.logger(f'Continuing from epoch {self.global_epoch}.')

        ''' Train '''
        # 中文：记录初始状态，后续每个 epoch 会追加到 _states 便于导出 dataframe
        self.max_epochs = self.global_epoch + num_epochs - 1
        self.dataframe = []
        self.state = {
            'train_loss': None,
            'train_metric': None,
            'train_monitor': None,
            'valid_loss': None,
            'valid_metric': None,
            'valid_monitor': None,
            'best_epoch': self.global_epoch,
            'best_score': None,
            'patience': 0,
            'epoch': 0,
            'learning_rate': [group['lr'] for group in self.optimizer.param_groups][0]
        }
        self._states = []

        if self.parallel == 'ddp':
            if self.xla:
                xmp.spawn(
                    self._train_xla,
                    args=(loader, loader_valid, num_epochs),
                    nprocs=self.world_size,
                    start_method='fork'
                )

            else:
                dist_url = f'tcp://127.0.0.1:{comm.find_free_port()}'
                session_id = str(uuid.uuid4())
                origin = Path.cwd() / __main__.__file__
                # 中文：使用 torch.distributed.launch 启动多进程并通过临时文件传参
                self.logger(f'DDP URL :\t{dist_url}')
                self.logger(f'session id :\t{session_id}')
                self.logger(f'__main__ :\t{origin}')
                
                ddp_tmp = {
                    'trainer': self,
                    'dist_url': dist_url,
                    'loader': loader,
                    'loader_valid': loader_valid,
                    'num_epochs': num_epochs
                }
                ddp_tmp_path = Path(f'.ku_ddp_tmp_{session_id}')
                with open(ddp_tmp_path, 'wb') as f:
                    pickle.dump(ddp_tmp, f)
                ddp_worker_path = Path(inspect.getfile(
                    self.__class__)).parent/'ddp_worker.py'
                env_copy = os.environ.copy()
                env_copy['OMP_NUM_THREADS'] = '1'
                command = [
                    'python',
                    '-m', 'torch.distributed.launch',
                    '--nproc_per_node', str(self.world_size), 
                    ddp_worker_path,
                    '--path', ddp_tmp_path,
                    '--origin', str(origin)
                ]
                proc = subprocess.Popen(
                    command, env=env_copy, cwd=origin.parent)
                proc.wait()
                if ddp_tmp_path.exists():
                    ddp_tmp_path.unlink()
        else:
            self._configure_model()
            self._train(loader, loader_valid, num_epochs)

    fit = train  # for compatibility # 为了兼容性，将 fit 方法指向 train 方法
    load_checkpoint = load_snapshot
    save_checkpoint = save_snapshot

    def export_dataframe(self):
        # 中文：将训练过程中累积的 state 列表转换为 DataFrame，便于分析/绘图
        return pd.DataFrame(self._states)

    def __repr__(self):
        # 中文：打印 Trainer 的核心信息，便于调试/日志
        print_dict = {
            'model': self.model.__class__.__name__,
            'device': self.device,
            'serial': self.serial
        }
        return f'TorchTrainer(\n{pformat(print_dict, compact=True, indent=2)})'
