import torch
import torch.nn as nn
import torch.nn.functional as F
# 确保以下组件路径在你的项目中正确
from models.backbones.visual_resnet import VisualBackbone
from models.backbones.structural import StructuralBackbone
from models.heads.projections import ProjectionHead
from models.heads.ppm import ViewAwarePPM
from models.decoders.fam_fusion import FAMFusion


class HPM(nn.Module):
    """
    水平金字塔映射 (Horizontal Pyramid Mapping)
    将特征图在高度维度切分为不同的条带（Bins），提取局部步态细节
    """

    def __init__(self, bin_list=[1, 2, 4, 8]):
        super(HPM, self).__init__()
        self.bin_list = bin_list

    def forward(self, x):
        # x 形状: [N, C, H, W]
        n, c, h, w = x.size()
        features = []
        for b in self.bin_list:
            # 沿高度 H 方向均匀切分为 b 个块
            z = F.adaptive_max_pool2d(x, (b, 1))  # [N, C, b, 1]
            features.append(z.view(n, -1))  # 展平 [N, C*b]
        return torch.cat(features, dim=-1)


class PAGFSLModel(nn.Module):
    def __init__(self, common_dim=512, lora_rank=16):
        super().__init__()
        self.debug_once = True  # 🌟 维度自检开关

        # 1. 视觉流: ResNet + HPM
        self.vis_backbone = VisualBackbone(lora_rank=lora_rank, pretrained=True)
        self.hpm = HPM(bin_list=[1, 2, 4, 8])
        # 假设 ResNet 输出通道为 512 (如 ResNet18/34)
        vis_hpm_dim = 512 * sum([1, 2, 4, 8])

        # 2. 结构流: DSTformer (LoRA 增强版)
        self.struct_backbone = StructuralBackbone(dim_rep=common_dim)

        # 3. 聚合层: 视角感知 PPM
        self.ppm = ViewAwarePPM(feature_dim=common_dim, num_views=11)

        # 4. 投影头: 统一维度到 512
        self.proj_vis = ProjectionHead(input_dim=vis_hpm_dim, common_dim=common_dim)
        self.proj_struct = ProjectionHead(input_dim=common_dim, common_dim=common_dim)

        # 5. 融合模块: FAM
        self.fam = FAMFusion(common_dim)

    def forward_feature(self, x_vis, x_struct, phase_weights, batch_size=32, view_idx=None):
        """
        x_vis: [B, T, C, H, W]
        x_struct: [B, T, 51]
        view_idx: [B] 🌟 必须接收视角索引
        """
        B, T, C, H, W = x_vis.shape

        if self.debug_once:
            print(f"\n🛡️ [维度自检] 输入 x_vis: {x_vis.shape}, x_struct: {x_struct.shape}")

        # --- A. 视觉流 ---
        x_vis_flat = x_vis.view(B * T, C, H, W)
        features_vis = []

        with torch.amp.autocast('cuda'):
            for i in range(0, B * T, batch_size):
                x_batch = x_vis_flat[i:i + batch_size]
                f_map = self.vis_backbone(x_batch)  # [batch, 512, H', W']
                f_hpm = self.hpm(f_map)  # [batch, 7680]
                features_vis.append(f_hpm.float())

        f_vis_flat = torch.cat(features_vis, dim=0)
        f_vis_seq = f_vis_flat.view(B, T, -1)
        f_vis_pool = f_vis_seq.max(dim=1)[0]  # 时序池化
        f_vis = self.proj_vis(f_vis_pool)  # 降维到 512

        # --- B. 结构流 ---
        # 🌟 关键：将 view_idx 传给支路进行视角补偿
        f_struct_seq = self.struct_backbone(x_struct, view_idx=view_idx)
        f_struct_final = self.ppm(f_struct_seq, phase_weights, view_idx=view_idx)
        f_struct = self.proj_struct(f_struct_final)

        if self.debug_once:
            print(f"🛡️ [维度自检] 视觉特征: {f_vis.shape}, 结构特征: {f_struct.shape}")

        # --- C. 融合 ---
        f_final = self.fam(f_vis, f_struct)

        if self.debug_once:
            print(f"🛡️ [维度自检] 融合特征: {f_final.shape}\n")
            self.debug_once = False  # 仅打印一次

        return f_final, f_vis, f_struct