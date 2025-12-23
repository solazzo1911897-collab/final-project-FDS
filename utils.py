import os
import random
import subprocess
import numpy as np
import torch
import time
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA = True
except ModuleNotFoundError:
    XLA = False


def freeze_module(module):
    # 中文：冻结模型参数，关闭梯度计算（常用于微调时固定 backbone）
    for i, param in enumerate(module.parameters()):
        param.requires_grad = False


def fit_state_dict(state_dict, model):
    '''
    Ignore size mismatch when loading state_dict
    '''
    # 中文：加载预训练权重时，遇到形状不匹配的参数直接跳过（打印提示）
    for name, param in model.named_parameters():
        if name in state_dict.keys():
            new_param = state_dict[name]
        else:
            continue
        if new_param.size() != param.size():
            print(f'Size mismatch in {name}: {new_param.shape} -> {param.shape}')
            state_dict.pop(name)


def get_device(arg):
    # 中文：根据传入参数决定计算设备与 device_ids：
    # - torch.device / xla_device：直接使用
    # - None / list / tuple：自动选择可用 GPU，否则 CPU；XLA 优先
    # - 字符串：直接转为 torch.device（'xla' 需环境支持 XLA）
    if isinstance(arg, torch.device) or \
        (XLA and isinstance(arg, xm.xla_device)):
        device = arg
    elif arg is None or isinstance(arg, (list, tuple)):
        if XLA:
            device = xm.xla_device()
        else:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(arg, str):
        if arg == 'xla' and XLA:
            device = xm.xla_device()
        else:
            device = torch.device(arg)
    
    if isinstance(arg, (list, tuple)):
        if isinstance(arg[0], int):
            device_ids = list(arg)
        elif isinstance(arg[0], str) and arg[0].isnumeric():
             device_ids = [ int(a) for a in arg ]
        else:
            raise ValueError(f'Invalid device: {arg}')
    else:
        if device.type == 'cuda':
            assert torch.cuda.is_available()
            if device.index is None:
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    device_ids = list(range(device_count))
                else:
                    device_ids = [0]
            else:
                device_ids = [device.index]
        else:
            device_ids = [device.index]
    
    return device, device_ids


def seed_everything(random_state=0, deterministic=False):
    """
    设置所有随机数生成器的种子，确保实验可复现
    
    Args:
        random_state (int): 随机种子值
        deterministic (bool): 是否使用确定性算法
            - True: 使用确定性算法，确保完全可复现（相同输入产生相同输出）
                  * 设置 cudnn.deterministic = True：cuDNN 使用确定性算法（可能更慢）
                  * 设置 cudnn.benchmark = False：禁用 cuDNN 的自动优化（可能更慢）
                  * 适用于需要完全可复现结果的场景（如论文实验、调试）
            - False（默认）: 允许非确定性算法，提升性能
                  * 设置 cudnn.deterministic = False：允许 cuDNN 使用非确定性算法（更快）
                  * cuDNN 会自动选择最优算法，提升训练速度
                  * 适用于生产环境，追求训练速度的场景
    
    注意：
        - 即使设置 deterministic=True，由于 GPU 并行计算的特性，完全可复现仍可能难以保证
        - deterministic=True 会显著降低训练速度（可能慢 10-30%），因为禁用了 cuDNN 的优化
        - 对于大多数场景，deterministic=False 是更好的选择（在可复现性和性能之间取得平衡）
    
    示例：
        # 完全可复现（较慢）
        seed_everything(42, deterministic=True)
        
        # 基本可复现（较快，推荐）
        seed_everything(42, deterministic=False)
    """
    # 设置 Python 标准库的随机数生成器
    random.seed(random_state)
    # 设置 Python 哈希随机化种子（影响字典、集合等数据结构的迭代顺序）
    os.environ['PYTHONHASHSEED'] = str(random_state)
    # 设置 NumPy 的随机数生成器
    np.random.seed(random_state)
    # 设置 PyTorch CPU 的随机数生成器
    torch.manual_seed(random_state)
    # 设置 PyTorch CUDA 的随机数生成器（所有 GPU）
    torch.cuda.manual_seed(random_state)
    
    if deterministic:
        # 确定性模式：确保 cuDNN 使用确定性算法（完全可复现，但较慢）
        torch.backends.cudnn.deterministic = True
        # 禁用 cuDNN 的自动优化（benchmark），确保算法选择的一致性
        torch.backends.cudnn.benchmark = False
    else:
        # 非确定性模式：允许 cuDNN 使用非确定性算法（更快，但可能略有差异）
        torch.backends.cudnn.deterministic = False
        # 注意：benchmark 保持默认值（通常为 False），由 PyTorch 自动管理


def get_gpu_memory():
    """
    Code borrowed from: 
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_time(time_format='%H:%M:%S'):
    # 中文：返回当前本地时间的格式化字符串
    return time.strftime(time_format, time.localtime())





from pprint import pformat
import types
import os
import requests


def print_config(cfg, logger=None):
    """打印配置对象中的关键字段，支持写入外部 logger。"""

    def _print(text):
        if logger is None:
            print(text)
        else:
            logger(text)
    
    items = [
        'name', 
        'cv', 'num_epochs', 'batch_size', 'seed',
        'dataset', 'dataset_params', 'num_classes', 'transforms', 'splitter',
        'model', 'model_params', 'weight_path', 'optimizer', 'optimizer_params',
        'scheduler', 'scheduler_params', 'batch_scheduler', 'scheduler_target',
        'criterion', 'eval_metric', 'monitor_metrics',
        'amp', # 'parallel', 
        'hook', 'callbacks', 'deterministic', 
        # 'clip_grad', 'max_grad_norm',
        'pseudo_labels'
    ]
    _print('===== CONFIG =====')
    for key in items:
        try:
            val = getattr(cfg, key)
            if isinstance(val, (type, types.FunctionType)):
                val = val.__name__ + '(*)'
            if isinstance(val, (dict, list)):
                val = '\n'+pformat(val, compact=True, indent=2)
            _print(f'{key}: {val}')
        except:
            _print(f'{key}: ERROR')
    _print(f'===== CONFIGEND =====')



class CallbackTemplate:
    '''
    Callback is called before or after each epoch.
    '''

    def __init__(self):
        pass

    def before_epoch(self, env):
        pass

    def after_epoch(self, env):
        pass

    def save_snapshot(self, trainer, path):
        pass

    def load_snapshot(self, trainer, path, device):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return self.__class__.__name__




class SaveEveryEpoch(CallbackTemplate):
    '''
    Save snapshot every epoch
    '''

    def __init__(self, patience=5, target='valid_metric', maximize=False, skip_epoch=0):
        super().__init__()
        
    def after_epoch(self, env):
        env.checkpoint = True
    

class EarlyStopping(CallbackTemplate):
    '''
    Early stops the training if validation loss doesn't improve after a given patience.
    patience: int       = 
    target: str         = 
    maximize: bool      = 
    skip_epoch: int     =
    '''

    def __init__(self, patience=5, target='valid_metric', maximize=False, skip_epoch=0):
        super().__init__()
        self.state = {
            'patience': patience,
            'target': target,
            'maximize': maximize,
            'skip_epoch': skip_epoch,
            'counter': 0,
            'best_score': None,
            'best_epoch': None
        }

    def after_epoch(self, env):
        score = env.state[self.state['target']]
        epoch = env.state['epoch'] # local epoch
        if epoch < self.state['skip_epoch'] or epoch == 0:
            self.state['best_score'] = score
            self.state['best_epoch'] = env.global_epoch
            env.checkpoint = True
            env.state['best_score'] = self.state['best_score']
            env.state['best_epoch'] = self.state['best_epoch']
        else:
            if (self.state['maximize'] and score > self.state['best_score']) or \
                    (not self.state['maximize'] and score < self.state['best_score']):
                self.state['best_score'] = score
                self.state['best_epoch'] = env.global_epoch
                self.state['counter'] = 0
                env.checkpoint = True
                env.state['best_score'] = self.state['best_score']
                env.state['best_epoch'] = self.state['best_epoch']
            else:
                self.state['counter'] += 1

            env.state['patience'] = self.state['counter']
            if self.state['counter'] >= self.state['patience']:
                env.stop_train = True

    def state_dict(self):
        return self.state

    def load_state_dict(self, checkpoint):
        self.state = checkpoint

    def __repr__(self):
        return f'EarlyStopping(\n{pformat(self.state, compact=True, indent=2)})'





"""
简单的训练钩子实现

TrainHook 是训练过程中最常用的钩子，提供了标准的训练、验证和测试流程。
它负责定义模型的前向传播、损失计算和评估指标的计算方式。
"""


class HookTemplate:
    '''
    Hook is called in each mini-batch during traing / inference 
    and after processed all mini-batches, 
    in order to define the training process and evaluate the results of each epoch.
    '''

    def __init__(self):
        pass

    def forward_train(self, trainer, inputs):
        # return loss, approx
        pass

    forward_valid = forward_train

    def forward_test(self, trainer, inputs, approx):
        # return approx
        pass

    def evaluate_batch(self, trainer, inputs):
        # return None
        pass

    def evaluate_epoch(self, trainer):
        # return metric_total, monitor_metrics_total
        pass



class TrainHook(HookTemplate):
    """
    训练钩子类 - 实现标准的训练流程
    
    主要功能：
    1. 定义训练/验证/测试时的前向传播逻辑
    2. 计算损失函数
    3. 计算评估指标（主指标和监控指标）
    4. 支持两种评估模式：
       - evaluate_in_batch=False（默认）：在每个 batch 收集预测和目标，epoch 结束时统一计算指标
       - evaluate_in_batch=True：在每个 batch 就计算指标，epoch 结束时取平均
    """

    def __init__(self, evaluate_in_batch=False):
        """
        初始化训练钩子
        
        Args:
            evaluate_in_batch (bool): 是否在每个 batch 就计算指标
                - False（默认）：收集所有 batch 的预测和目标，epoch 结束时统一计算（更准确，但占用更多内存）
                - True：每个 batch 计算指标，epoch 结束时取平均（更快，但可能不够准确）
        """
        super().__init__()
        self.evaluate_in_batch = evaluate_in_batch

    def _evaluate(self, trainer, approx, target):
        """
        计算评估指标的内部方法
        
        Args:
            trainer: TorchTrainer 实例，包含 eval_metric 和 monitor_metrics
            approx: 模型预测值（可以是单个 batch 或整个 epoch 的预测）
            target: 真实标签（可以是单个 batch 或整个 epoch 的标签）
        
        Returns:
            tuple: (metric_score, monitor_score)
                - metric_score: 主评估指标得分（如 AUC），如果未设置则为 None
                - monitor_score: 监控指标得分列表（如 F1, Precision, Recall 等）
        """
        if trainer.eval_metric is None:
            metric_score = None
        else:
            metric_score = trainer.eval_metric(approx, target)
        monitor_score = []
        for monitor_metric in trainer.monitor_metrics:
            monitor_score.append(
                monitor_metric(approx, target))
        return metric_score, monitor_score

    def forward_train(self, trainer, inputs):
        """
        训练时的前向传播
        
        Args:
            trainer: TorchTrainer 实例
            inputs: 输入数据元组，最后一个元素是标签，前面是模型输入
        
        Returns:
            tuple: (loss, approx_detached)
                - loss: 损失值（用于反向传播）
                - approx_detached: 模型预测值（detached，不参与梯度计算，用于评估）
        """
        target = inputs[-1]  # 最后一个元素是标签
        approx = trainer.model(*inputs[:-1])  # 前面的元素是模型输入
        loss = trainer.criterion(approx, target)  # 计算损失
        return loss, approx.detach()  # 返回损失和预测值（detached）

    forward_valid = forward_train  # 验证时的前向传播与训练相同

    def forward_test(self, trainer, inputs):
        """
        测试时的前向传播（无标签，只返回预测）
        
        Args:
            trainer: TorchTrainer 实例
            inputs: 输入数据元组（不包含标签）
        
        Returns:
            approx: 模型预测值
        """
        approx = trainer.model(*inputs[:-1])  # 测试时 inputs 可能仍包含占位符，取前面的元素
        return approx

    def evaluate_batch(self, trainer, inputs, approx):
        """
        批次级别的评估
        
        根据 evaluate_in_batch 设置，有两种模式：
        1. evaluate_in_batch=True：立即计算指标并存储
        2. evaluate_in_batch=False：只存储预测和目标，epoch 结束时统一计算
        
        Args:
            trainer: TorchTrainer 实例
            inputs: 输入数据元组
            approx: 模型预测值（已通过前向传播得到）
        """
        target = inputs[-1]  # 获取标签
        storage = trainer.epoch_storage  # 获取 epoch 存储字典
        if self.evaluate_in_batch:
            # 模式1：立即计算指标并存储（每个 batch 计算一次）
            metric_score, monitor_score = self._evaluate(trainer, approx, target)
            storage['batch_metric'].append(metric_score)
            storage['batch_monitor'].append(monitor_score)
        else:
            # 模式2：只存储预测和目标（epoch 结束时统一计算，更准确）
            storage['approx'].append(approx)
            storage['target'].append(target)

    def evaluate_epoch(self, trainer):
        """
        Epoch 级别的评估
        
        根据 evaluate_in_batch 设置，有两种计算方式：
        1. evaluate_in_batch=True：对所有 batch 的指标取平均
        2. evaluate_in_batch=False：使用收集的所有预测和目标统一计算指标
        
        Args:
            trainer: TorchTrainer 实例
        
        Returns:
            tuple: (metric_total, monitor_total)
                - metric_total: 主评估指标得分（整个 epoch 的）
                - monitor_total: 监控指标得分列表（整个 epoch 的）
        """
        storage = trainer.epoch_storage
        if self.evaluate_in_batch:
            # 模式1：对所有 batch 的指标取平均
            metric_total = storage['batch_metric'].mean(0)
            monitor_total = storage['batch_monitor'].mean(0).tolist()

        else: 
            # 模式2：使用收集的所有预测和目标统一计算指标（更准确）
            metric_total, monitor_total = self._evaluate(
                trainer, storage['approx'], storage['target'])
        return metric_total, monitor_total


SimpleHook = TrainHook  # 兼容性别名，保持向后兼容





from torch.nn.parallel import DataParallel, DistributedDataParallel


def _save_snapshot(trainer, path, 
                   save_optimizer=False, 
                   save_scheduler=False):
    if isinstance(
            trainer.model,
            (DataParallel, DistributedDataParallel)):
        module = trainer.model.module
    else:
        module = trainer.model
    
    serialized = {
        'global_epoch': trainer.global_epoch,
        'model': module.state_dict(),
        'state': trainer.state,
        'all_states': trainer._states
    }
    if save_optimizer:
        serialized['optimizer'] = trainer.optimizer.state_dict()
    if save_scheduler:
        serialized['scheduler'] = trainer.scheduler.state_dict()

    if trainer.xla:
        import torch_xla.utils.serialization as xser
        xser.save(serialized, str(path))
    else:
        torch.save(serialized, str(path))


def _load_snapshot(trainer, path, device):
    if trainer.xla:
        import torch_xla.utils.serialization as xser
        checkpoint = xser.load(str(path))
    else:
        checkpoint = torch.load(str(path), map_location=device)

    if isinstance(
            trainer.model,
            (DataParallel, DistributedDataParallel)):
        trainer.model.module.load_state_dict(checkpoint['model'])
    else:
        trainer.model.load_state_dict(checkpoint['model'])

    if hasattr(trainer, 'optimizer') and 'optimizer' in checkpoint.keys():
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    if hasattr(trainer, 'scheduler') and 'scheduler' in checkpoint.keys():
        trainer.scheduler.load_state_dict(checkpoint['scheduler'])
    if hasattr(trainer, 'global_epoch'):
        trainer.global_epoch = checkpoint['global_epoch']
    trainer.state = checkpoint['state']
    trainer._states = checkpoint['all_states']


class SaveSnapshot(CallbackTemplate):
    '''
    Path priority: path argument > BestEpoch.path > trainer.snapshot_path
    '''

    def __init__(self, path=None, save_optimizer=False, save_scheduler=False):
        super().__init__()
        self.path = path
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

    def save_snapshot(self, trainer, path):
        if path is None:
            path = self.path if self.path is not None else trainer.snapshot_path
        _save_snapshot(trainer, path, self.save_optimizer, self.save_scheduler)

    def load_snapshot(self, trainer, path=None, device=None):
        if path is None:
            path = self.path if self.path is not None else trainer.snapshot_path
        if device is None:
            device = trainer.device
        _load_snapshot(trainer, path, device)