"""
评估指标基类模板

本模块提供了一个通用的评估指标基类，支持多种机器学习框架。
通过继承 MetricTemplate 并实现 _test 方法，可以轻松创建自定义评估指标。

支持的框架：
- 通用 Python：直接调用 Metric()(target, approx)
- CatBoost：eval_metric=Metric()
- LightGBM：metric='Metric_Name', feval=Metric().lgb
- XGBoost：使用 Metric().xgb
- PyTorch：Metric().torch(output, labels)
"""
import numpy as np
import pandas as pd


class MetricTemplate:
    """
    评估指标基类模板
    
    所有自定义评估指标都应该继承此类，并实现 _test 方法来定义具体的计算逻辑。
    
    使用示例：
        # 通用用法
        metric = AUC()
        score = metric(target, approx)
        
        # CatBoost
        model = CatBoostClassifier(eval_metric=AUC())
        
        # LightGBM
        model = lgb.train(params, train_data, feval=AUC().lgb)
        
        # PyTorch
        metric = AUC()
        score = metric.torch(model_output, labels)
    """

    def __init__(self, maximize=False):
        """
        初始化评估指标
        
        Args:
            maximize (bool): 该指标是否越大越好
                - True: 指标越大越好（如 AUC、准确率），用于早停、模型选择等
                - False: 指标越小越好（如损失、MAE），用于早停、模型选择等
        """
        self.maximize = maximize

    def __repr__(self):
        """
        返回指标对象的字符串表示
        
        Returns:
            str: 格式为 "MetricName(maximize=True/False)"
        """
        return f'{type(self).__name__}(maximize={self.maximize})'

    def _test(self, target, approx):
        """
        计算评估指标的核心方法（需要在子类中实现）
        
        这是所有评估指标的核心逻辑，子类必须实现此方法。
        
        Args:
            target: 真实标签（ground truth）
            approx: 模型预测值
        
        Returns:
            float: 评估指标的值
        
        注意：
            子类必须重写此方法来实现具体的指标计算逻辑
        """
        # Metric calculation
        pass

    def __call__(self, target, approx):
        """
        使指标对象可调用（通用接口）
        
        允许像函数一样使用指标对象：metric(target, approx)
        
        Args:
            target: 真实标签
            approx: 模型预测值
        
        Returns:
            float: 评估指标的值
        """
        return self._test(target, approx)

    # ===== CatBoost 接口 =====
    def get_final_error(self, error, weight):
        """
        CatBoost 接口：计算最终误差
        
        CatBoost 在训练过程中会调用此方法来计算最终的评估指标值。
        
        Args:
            error: 累计误差值
            weight: 权重总和
        
        Returns:
            float: 最终误差值（error / weight）
        """
        return error / weight

    def is_max_optimal(self):
        """
        CatBoost 接口：判断指标是否越大越好
        
        CatBoost 使用此方法来判断在早停、模型选择等场景中，应该最大化还是最小化该指标。
        
        Returns:
            bool: True 表示越大越好，False 表示越小越好
        """
        return self.maximize

    def evaluate(self, approxes, target, weight=None):
        """
        CatBoost 接口：评估方法
        
        CatBoost 在训练过程中会调用此方法来计算评估指标。
        
        Args:
            approxes: 预测值列表，每个元素是一个列表（支持多输出模型）
                     对于单输出模型，approxes[0] 是预测值
            target: 真实标签（列表或数组）
            weight: 样本权重（可选），如果为 None 则使用均匀权重
        
        Returns:
            tuple: (error_sum, weight_sum)
                - error_sum: 评估指标值（对于最大化指标，这是得分；对于最小化指标，这是误差）
                - weight_sum: 权重总和（通常为 1.0）
        
        注意：
            - CatBoost 要求返回 (error, weight) 元组
            - 对于最大化指标（如 AUC），error_sum 实际上是得分
            - 对于最小化指标（如损失），error_sum 是误差值
        """
        # approxes - list of list-like objects (one object per approx dimension)
        # target - list-like object
        # weight - list-like object, can be None
        assert len(approxes[0]) == len(target)
        if not isinstance(target, np.ndarray):
            target = np.array(target)

        approx = np.array(approxes[0])
        error_sum = self._test(target, approx)
        weight_sum = 1.0

        return error_sum, weight_sum

    # ===== LightGBM 接口 =====
    def lgb(self, approx, data):
        """
        LightGBM 接口：评估方法
        
        LightGBM 在训练过程中会调用此方法来计算评估指标。
        
        Args:
            approx: 模型预测值（numpy 数组）
            data: LightGBM Dataset 对象，包含标签等信息
        
        Returns:
            tuple: (metric_name, metric_value, is_higher_better)
                - metric_name: 指标名称（类名）
                - metric_value: 指标值
                - is_higher_better: 是否越大越好（用于早停等）
        
        使用示例：
            model = lgb.train(
                params,
                train_data,
                feval=AUC().lgb,  # 使用自定义评估指标
                ...
            )
        """
        target = data.get_label()
        return self.__class__.__name__, self._test(target, approx), self.maximize

    lgbm = lgb  # 兼容性别名，保持向后兼容

    # ===== XGBoost 接口 =====
    def xgb(self, approx, dtrain):
        """
        XGBoost 接口：评估方法
        
        XGBoost 在训练过程中会调用此方法来计算评估指标。
        
        Args:
            approx: 模型预测值（numpy 数组）
            dtrain: XGBoost DMatrix 对象，包含标签等信息
        
        Returns:
            tuple: (metric_name, metric_value)
                - metric_name: 指标名称（类名）
                - metric_value: 指标值
        
        使用示例：
            model = xgb.train(
                params,
                dtrain,
                feval=AUC().xgb,  # 使用自定义评估指标
                ...
            )
        """
        target = dtrain.get_label()
        return self.__class__.__name__, self._test(target, approx)

    # ===== PyTorch 接口 =====
    def torch(self, approx, target):
        """
        PyTorch 接口：评估方法
        
        将 PyTorch 张量转换为 numpy 数组后计算评估指标。
        自动处理梯度分离和 CPU 转换。
        
        Args:
            approx: 模型预测值（PyTorch 张量）
            target: 真实标签（PyTorch 张量）
        
        Returns:
            float: 评估指标的值
        
        注意：
            - 自动调用 detach() 分离梯度（不参与反向传播）
            - 自动转换到 CPU 和 numpy 数组
            - 适用于训练和验证过程中的指标计算
        
        使用示例：
            metric = AUC()
            output = model(inputs)
            score = metric.torch(output, labels)
        """
        return self._test(target.detach().cpu().numpy(),
                          approx.detach().cpu().numpy())









"""
评估指标模块

本模块定义了模型训练和验证过程中使用的评估指标。
AUC（Area Under ROC Curve）是二分类任务中最常用的评估指标之一。
"""
import numpy as np
from sklearn.metrics import roc_auc_score


class AUC(MetricTemplate):
    """
    ROC 曲线下面积（Area Under ROC Curve）评估指标
    
    AUC 是二分类任务中最重要的评估指标之一，表示模型区分正负样本的能力。
    
    特点：
    - AUC 值范围：[0, 1]
    - AUC = 0.5：模型性能等同于随机猜测
    - AUC = 1.0：完美分类器
    - AUC > 0.5：模型有区分能力，值越大越好
    
    适用场景：
    - 二分类任务（如引力波检测：有信号/无信号）
    - 类别不平衡的数据集（AUC 不受类别分布影响）
    - 需要评估模型整体排序能力的场景
    
    注意：
    - maximize=True 表示该指标越大越好（用于早停、模型选择等）
    - 支持多种输入格式：1D 向量、2D 单列、2D 两列（取正类概率）
    """
    def __init__(self):
        """
        初始化 AUC 评估指标
        
        设置 maximize=True，表示该指标越大越好（用于早停、模型选择等场景）
        """
        super().__init__(maximize=True)

    def _test(self, target, approx):
        """
        计算 AUC 值
        
        该方法处理不同格式的输入，并计算 ROC 曲线下面积。
        
        Args:
            target: 真实标签，形状为 (n_samples,) 或 (n_samples, 1)
                   - 值应为 0 或 1（二分类）
                   - 会自动四舍五入到最近的整数
            approx: 模型预测值，支持多种格式：
                   - 形状 (n_samples,): 1D 向量，直接使用
                   - 形状 (n_samples, 1): 2D 单列，压缩为 1D
                   - 形状 (n_samples, 2): 2D 两列（二分类），取第二列（正类概率）
        
        Returns:
            float: AUC 值，范围 [0, 1]
        
        Raises:
            ValueError: 如果 approx 的形状不符合预期
        
        示例：
            >>> auc = AUC()
            >>> target = np.array([0, 1, 1, 0])
            >>> approx = np.array([0.1, 0.9, 0.8, 0.2])  # 1D 格式
            >>> score = auc._test(target, approx)
            >>> # 或者
            >>> approx = np.array([[0.1], [0.9], [0.8], [0.2]])  # 2D 单列格式
            >>> score = auc._test(target, approx)
        """
        # 处理不同格式的预测值输入
        if len(approx.shape) == 1:
            # 情况1：已经是 1D 向量，直接使用（形状: (n_samples,)）
            pass
        elif approx.shape[1] == 1:
            # 情况2：2D 单列格式，压缩为 1D（形状: (n_samples, 1) -> (n_samples,)）
            # 例如：[[0.1], [0.9], [0.8]] -> [0.1, 0.9, 0.8]
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            # 情况3：2D 两列格式（二分类），取第二列作为正类概率（形状: (n_samples, 2) -> (n_samples,)）
            # 例如：[[0.9, 0.1], [0.2, 0.8]] -> [0.1, 0.8]
            # 第一列是负类概率，第二列是正类概率，我们使用正类概率
            approx = approx[:, 1]
        else:
            # 如果形状不符合预期，抛出错误
            raise ValueError(f'Invalid approx shape: {approx.shape}')
        
        # 将标签四舍五入到最近的整数（确保是 0 或 1）
        # 这可以处理浮点数标签（如 0.0, 1.0）或接近整数的值
        target = np.round(target)
        
        # 使用 sklearn 计算 ROC 曲线下面积
        # roc_auc_score 要求：
        # - target: 真实标签（0 或 1）
        # - approx: 预测概率或得分（值越大表示越可能是正类）
        return roc_auc_score(target, approx)
