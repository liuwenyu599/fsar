import torch
import torch.nn as nn
from models.backbones.visual import VisualBackbone
from models.backbones.structural import StructuralBackbone
from models.heads.projections import ProjectionHead
from models.heads.ppm import PPM
from models.decoders.fam_fusion import FAMFusion


class PAGFSLModel(nn.Module):
    def __init__(self, common_dim=512, input_vis=768, input_struct=51):
        super().__init__()

        # 1. 骨干网络
        # 视觉流：ViT + LoRA
        self.vis_backbone = VisualBackbone(input_vis, common_dim)
        # 结构流：MotionBERT
        self.struct_backbone = StructuralBackbone(input_struct, common_dim)

        # 2. 结构流强化 (PPM 模块)
        self.ppm = PPM()

        # 3. 投影头 (维度对齐 & 语义对齐)
        # 将两流特征映射到完全一致的维度 D
        self.proj_vis = ProjectionHead(common_dim, common_dim)
        self.proj_struct = ProjectionHead(common_dim, common_dim)

        # 4. 融合模块 (FAM)
        self.fam = FAMFusion(common_dim, common_dim, common_dim)

    def forward_feature(self, x_vis, x_struct, phase_weights):
        """
        前向传播逻辑: 返回融合特征和各流特征
        """
        # --- A. 视觉流 (Student) ---
        # 1. 骨干提取 (B, T, D)
        f_vis_raw = self.vis_backbone(x_vis)
        # 2. 简单时序聚合 (这里用平均，也可以加Attention)
        f_vis_pool = f_vis_raw.mean(dim=1)
        # 3. 投影 (F_vis) -> 用于 L_align 和 L_disc
        f_vis = self.proj_vis(f_vis_pool)

        # --- B. 结构流 (Teacher) ---
        # 1. 骨干提取 (B, T, D)
        f_struct_seq = self.struct_backbone(x_struct)
        # 2. PPM 物理强化 (使用 GPE 权重聚合)
        f_struct_ppm = self.ppm(f_struct_seq, phase_weights)
        # 3. 投影 (F_struct) -> 用于 L_align
        f_struct = self.proj_struct(f_struct_ppm)

        # --- C. 融合 (FAM) ---
        # F_final -> 用于 L_proto 分类
        f_final = self.fam(f_vis, f_struct)

        return f_final, f_vis, f_struct