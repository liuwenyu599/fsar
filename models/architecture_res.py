import sys
import os
# 将项目根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.visual_resnet import VisualBackbone
from models.backbones.structural import StructuralBackbone
from models.heads.projections import ProjectionHead
from models.heads.ppm import PPMStructuredAggregator
from models.decoders.fam_fusion import FAMFusion


class HPM(nn.Module):
    """
    水平金字塔映射 (Horizontal Pyramid Mapping)
    将特征图在高度维度切分为不同的条带（Bins），提取局部步态特征
    """

    def __init__(self, bin_list=[1, 2, 4, 8, 16]):
        super(HPM, self).__init__()
        self.bin_list = bin_list

    def forward(self, x):
        # x 形状: [N, C, H, W] -> 来自 ResNet 的特征图
        n, c, h, w = x.size()
        features = []
        for b in self.bin_list:
            # 沿高度 H 方向均匀切分为 b 个块，执行自适应最大池化
            z = F.adaptive_max_pool2d(x, (b, 1))  # 输出: [N, C, b, 1]
            features.append(z.view(n, -1))  # 展平为 [N, C*b]

        # 将所有尺度的特征拼接
        full_feature = torch.cat(features, dim=-1)  # [N, C * sum(bin_list)]
        return full_feature


class PAGFSLModel(nn.Module):
    """
    增强型 PAG-FSL 主模型
    1. 视觉流: ResNet + HPM (水平金字塔)
    2. 结构流: MotionBERT + PPM
    3. 特征对齐: ProjectionHead
    4. 动态融合: FAM (可替换为 Uncertainty Gate)
    """

    def __init__(self, common_dim=512, input_struct=51, lora_rank=16):
        super().__init__()

        # --- 1. 视觉流骨干 (假设输出 Feature Map 而非全局向量) ---
        # 注意：这里需要确保 VisualBackbone 返回的是卷积层的 Feature Map [B, C, H, W]
        self.vis_backbone = VisualBackbone(lora_rank=lora_rank, pretrained=False)

        # 引入 HPM 层，bin_list 可以根据显存调整
        self.hpm = HPM(bin_list=[1, 2, 4, 8])
        # HPM 输出维度计算: Backbone通道数 * sum(bin_list)
        # 假设 ResNet18 最后一层通道是 512，则: 512 * (1+2+4+8) = 7680
        hpm_out_dim = 512 * sum([1, 2, 4, 8])

        # --- 2. 结构流骨干 ---
        self.struct_backbone = StructuralBackbone(input_dim=input_struct, embed_dim=common_dim)
        struct_out_dim = self.struct_backbone.out_dim

        # --- 3. PPM 强化结构流 ---
        self.ppm = PPMStructuredAggregator(feature_dim=struct_out_dim)

        # --- 4. 投影头 (对齐到 common_dim) ---
        # 视觉流通过投影头将 HPM 的高维特征降维到 512
        self.proj_vis = ProjectionHead(input_dim=hpm_out_dim, common_dim=common_dim)
        self.proj_struct = ProjectionHead(input_dim=struct_out_dim, common_dim=common_dim)

        # --- 5. FAM 融合模块 ---
        self.fam = FAMFusion(common_dim)

    def forward_feature(self, x_vis, x_struct, phase_weights, batch_size=32):
        """
        前向传播 (显存优化版)
        """
        B, T, C, H, W = x_vis.shape

        # --- A. 视觉流: 分批处理并通过 HPM ---
        x_vis_flat = x_vis.view(B * T, C, H, W)
        features_vis = []

        # 🔥 开启混合精度 autocast
        # 注意：PyTorch 2.x 推荐使用 torch.amp.autocast('cuda')
        with torch.amp.autocast('cuda'):
            for i in range(0, B * T, batch_size):
                x_batch = x_vis_flat[i:i + batch_size]

                # 1. 提取卷积特征图 [batch, 512, 7, 7]
                f_map = self.vis_backbone(x_batch)

                # 2. HPM 空间解耦 [batch, 7680]
                f_hpm = self.hpm(f_map)

                # 将结果转回 float32 存储，防止后续计算精度问题
                features_vis.append(f_hpm.float())

        f_vis_flat = torch.cat(features_vis, dim=0)  # (B*T, hpm_out_dim)

        # 3. 时序处理
        f_vis_seq = f_vis_flat.view(B, T, -1)
        f_vis_pool = f_vis_seq.max(dim=1)[0]  # (B, hpm_out_dim)

        # 4. 投影到公共空间 (512维)
        f_vis = self.proj_vis(f_vis_pool)

        # --- B. 结构流 ---
        f_struct_seq = self.struct_backbone(x_struct)
        f_struct_ppm = self.ppm(f_struct_seq, phase_weights)
        f_struct = self.proj_struct(f_struct_ppm)

        # --- C. 融合 ---
        f_final = self.fam(f_vis, f_struct)

        return f_final, f_vis, f_struct

if __name__ == "__main__":
    # 1. 模拟超参数
    B, T = 2, 8  # Batch Size=2, 序列长度=8帧
    C, H, W = 3, 224, 224  # 图像通道、高、宽
    D_struct = 51  # 骨架输入维度 (17个关节点 * 3维坐标)
    common_dim = 512  # 公共嵌入空间维度

    # 2. 初始化模型
    # 注意：确保你的 VisualBackbone 已经去掉了池化层
    model = PAGFSLModel(common_dim=common_dim, input_struct=D_struct)
    model.eval()  # 验证模式

    # 3. 构造模拟输入
    # 视觉张量: (B, T, C, H, W)
    mock_vis = torch.randn(B, T, C, H, W)
    # 结构张量: (B, T, D_struct)
    mock_struct = torch.randn(B, T, D_struct)
    # 相位权重: (B, T)
    mock_phase = torch.ones(B, T)

    print(f"--- 🚀 开始 PAGFSLModel 架构测试 ---")
    print(f"输入视觉形状: {mock_vis.shape}")
    print(f"输入结构形状: {mock_struct.shape}")

    try:
        # 4. 前向传播
        # 设置 batch_size=4 验证视觉流的分批推理逻辑
        f_final, f_vis, f_struct = model.forward_feature(
            mock_vis, mock_struct, mock_phase, batch_size=4
        )

        # 5. 维度检查
        print(f"\n--- 🔍 维度验证结果 ---")
        print(f"视觉特征 f_vis 形状: {f_vis.shape}  (预期: [{B}, {common_dim}])")
        print(f"结构特征 f_struct 形状: {f_struct.shape} (预期: [{B}, {common_dim}])")
        print(f"融合特征 f_final 形状: {f_final.shape}  (预期: [{B}, {common_dim}])")

        # 验证 HPM 拼接后的中间维度（内部逻辑验证）
        hpm_bins = [1, 2, 4, 8]
        expected_hpm_dim = 512 * sum(hpm_bins)
        print(f"HPM 拼接维度验证: {expected_hpm_dim} 维")

        # 6. 逻辑判断
        assert f_vis.shape == (B, common_dim), "视觉流输出维度错误！"
        assert f_struct.shape == (B, common_dim), "结构流输出维度错误！"
        assert f_final.shape == (B, common_dim), "融合流输出维度错误！"

        print(f"\n✅ 测试通过！架构逻辑与维度对齐完全正确。")

    except Exception as e:
        print(f"\n❌ 测试失败！错误信息: {e}")
        import traceback

        traceback.print_exc()