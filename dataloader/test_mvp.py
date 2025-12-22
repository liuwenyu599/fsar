import sys
sys.path.append("/home/lwy/projects/fsan/motionbert/MotionBERT-main")

import torch
import numpy as np
from lib.model.DSTformer import DSTformer


def test_ppgait_mvp():
    print(">>> PPGait MVP 验证: OpenPose -> MotionBERT 适配测试 <<<")

    # ==========================================
    # 1. 模拟你的输入数据 (假设是 CASIA-B/OpenPose 格式)
    # OpenPose COCO (18点) 常见顺序:
    # 0:Nose, 1:Neck, 2:R-Sho, 3:Elbow, 4:Wrist, 5:L-Sho, 6:Elbow, 7:Wrist,
    # 8:R-Hip, 9:Knee, 10:Ankle, 11:L-Hip, 12:Knee, 13:Ankle, 14:R-Eye, 15:L-Eye, 16:R-Ear, 17:L-Ear
    batch_size = 2
    frames = 64
    my_openpose_data = torch.randn(batch_size, frames, 18, 3)
    print(f"1. 原始数据 (OpenPose): {my_openpose_data.shape}")

    # ==========================================
    # 2. 核心映射函数：OpenPose -> H36M 17点
    def openpose_to_h36m(data_18pt):
        B, F, J, C = data_18pt.shape
        new_data = torch.zeros(B, F, 17, C, device=data_18pt.device)

        nose = data_18pt[:, :, 0]
        neck = data_18pt[:, :, 1]
        r_sho, l_sho = data_18pt[:, :, 2], data_18pt[:, :, 5]
        r_elb, l_elb = data_18pt[:, :, 3], data_18pt[:, :, 6]
        r_wri, l_wri = data_18pt[:, :, 4], data_18pt[:, :, 7]
        r_hip, l_hip = data_18pt[:, :, 8], data_18pt[:, :, 11]
        r_kne, l_kne = data_18pt[:, :, 9], data_18pt[:, :, 12]
        r_ank, l_ank = data_18pt[:, :, 10], data_18pt[:, :, 13]

        # 关键计算
        pelvis = (r_hip + l_hip) / 2.0
        spine = (pelvis + neck) / 2.0
        head = nose + (nose - neck)

        # 赋值到 H36M 顺序
        new_data[:, :, 0] = pelvis
        new_data[:, :, 1] = r_hip
        new_data[:, :, 2] = r_kne
        new_data[:, :, 3] = r_ank
        new_data[:, :, 4] = l_hip
        new_data[:, :, 5] = l_kne
        new_data[:, :, 6] = l_ank
        new_data[:, :, 7] = spine
        new_data[:, :, 8] = neck
        new_data[:, :, 9] = nose
        new_data[:, :, 10] = head
        new_data[:, :, 11] = l_sho
        new_data[:, :, 12] = l_elb
        new_data[:, :, 13] = l_wri
        new_data[:, :, 14] = r_sho
        new_data[:, :, 15] = r_elb
        new_data[:, :, 16] = r_wri

        return new_data

    try:
        mapped_input = openpose_to_h36m(my_openpose_data)
        print(f"2. 映射完成 (H36M 17点): {mapped_input.shape}")
    except Exception as e:
        print(f"❌ 映射逻辑报错: {e}")
        return

    # ==========================================
    # 3. Root-Centering
    mapped_input = mapped_input - mapped_input[:, :, 0:1, :]
    print("3. 已执行 Root-Centering (以骨盆为原点)")

    # ==========================================
    # 4. 测试喂给 MotionBERT
    try:
        model = DSTformer(dim_in=3, dim_out=512, dim_feat=512, num_joints=17)
        output = model.get_representation(mapped_input)
        print("\n>>> 结果判定 <<<")
        print(f"✅ Forward 成功！输出特征: {output.shape}")

        if torch.isnan(output).any():
            print("❌ 失败：输出含 NaN")
        else:
            print("✅ 数值正常，接口打通！")
            print("🎉 下一步：把 openpose_to_h36m 封装到 Dataset 类里")

    except Exception as e:
        print(f"❌ Forward 报错: {e}")


if __name__ == "__main__":
    test_ppgait_mvp()
