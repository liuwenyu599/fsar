import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """
    特征对齐损失 (Alignment Loss / Knowledge Distillation)

    目标:
    强制学生网络 (Visual Stream) 的特征向教师网络 (Structure Stream) 靠拢。
    将结构流的“衣着不变性”和“物理鲁棒性”蒸馏给视觉流。

    公式: L_align = || F_vis - F_struct ||^2
    """

    def __init__(self):
        super(AlignmentLoss, self).__init__()
        # 使用均方误差 (MSE) 作为距离度量
        self.mse_loss = nn.MSELoss()

    def forward(self, f_student, f_teacher):
        """
        Args:
            f_student: (B, D) - 投影后的视觉特征 (F_vis)
            f_teacher: (B, D) - 投影后的结构特征 (F_struct)
        Returns:
            loss: 标量损失
        """
        # 1. 维度与形状检查
        if f_student.shape != f_teacher.shape:
            raise ValueError(f"Shape Mismatch for Alignment: Student {f_student.shape} vs Teacher {f_teacher.shape}")

        # 2. 特征归一化 (可选优化)
        # 在度量学习中，通常特征都在超球面上，归一化有助于稳定对齐
        # 这里我们先进行 L2 归一化，再计算 MSE，这等价于 Cosine Distance 的变体
        # 如果您的 Projection Head 最后没有 LayerNorm，建议开启此选项
        f_student_norm = F.normalize(f_student, p=2, dim=1)
        f_teacher_norm = F.normalize(f_teacher, p=2, dim=1)

        # 3. 计算 MSE 损失
        loss = self.mse_loss(f_student_norm, f_teacher_norm)

        return loss