import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn

class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]

def eval_metrics(output, target, num_class=2):
    """
    为二分类优化的评估指标计算

    Args:
        output: 模型输出 (B, C, H, W)
        target: 真实标签 (B, H, W)
        num_class: 类别数（二分类默认为2）

    Returns:
        [正确像素数, 总标签像素数, 交集, 并集]
    """
    # 对于二分类，使用sigmoid或softmax
    if output.shape[1] > 1:
        # 如果是多通道输出，取最大值
        predict = torch.argmax(output, dim=1)
    else:
        # 对于单通道输出，使用阈值分类
        predict = (output.squeeze(1) > 0.5).long()

    # 确保预测和目标在同一数据类型
    predict = predict.long()
    target = target.long()

    # 创建标签掩码（排除忽略的像素）
    labeled = (target >= 0) * (target < num_class)

    # 计算像素准确率
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)

    # 计算交集和并集
    inter, union = batch_intersection_union(predict, target, num_class, labeled)

    return [
        np.round(correct, 5),
        np.round(num_labeled, 5),
        np.round(inter, 5),
        np.round(union, 5)
    ]

def batch_pix_accuracy(predict, target, labeled):
    """
    计算像素准确率

    Args:
        predict: 预测结果
        target: 真实标签
        labeled: 有效像素掩码

    Returns:
        正确像素数, 总标签像素数
    """
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"

    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):
    """
    计算交集和并集

    Args:
        predict: 预测结果
        target: 真实标签
        num_class: 类别数
        labeled: 有效像素掩码

    Returns:
        每个类别的交集, 每个类别的并集
    """
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    # 对于二分类，可能需要特殊处理
    area_inter = torch.zeros(num_class, dtype=torch.float)
    area_pred = torch.zeros(num_class, dtype=torch.float)
    area_lab = torch.zeros(num_class, dtype=torch.float)

    for cls in range(num_class):
        area_inter[cls] = ((predict == cls) * (target == cls)).sum()
        area_pred[cls] = (predict == cls).sum()
        area_lab[cls] = (target == cls).sum()

    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def initialize_weights(*models):
    """
    权重初始化函数保持不变
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
