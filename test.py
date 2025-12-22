import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# 修改为你的实际路径
DATA_DIR = "/datasets/CASIA-B/pose/CASIA-B_HRNet"


# H36M 的连线定义 (用于画图)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]


def openpose_to_h36m(data_input):
    """
    自适应适配器：支持 COCO-17 (HRNet默认) 和 OpenPose-18
    输出: [T, 17, C] (H36M 格式)
    """
    T, J, C = data_input.shape
    new_data = np.zeros((T, 17, C), dtype=np.float32)

    # === 关键分支：判断输入数据格式 ===
    if J == 17:
        # COCO 17点格式 (CASIA-B HRNet 常用)
        # 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear,
        # 5:L_Sho, 6:R_Sho, 7:L_Elb, 8:R_Elb, 9:L_Wri, 10:R_Wri,
        # 11:L_Hip, 12:R_Hip, 13:L_Kne, 14:R_Kne, 15:L_Ank, 16:R_Ank
        print("ℹ️ 检测到 COCO 17点格式，正在计算 Neck...")

        nose = data_input[:, 0]
        # COCO没有脖子，必须通过左右肩中点计算
        l_sho, r_sho = data_input[:, 5], data_input[:, 6]
        neck = (l_sho + r_sho) / 2.0

        l_elb, r_elb = data_input[:, 7], data_input[:, 8]
        l_wri, r_wri = data_input[:, 9], data_input[:, 10]
        l_hip, r_hip = data_input[:, 11], data_input[:, 12]
        l_kne, r_kne = data_input[:, 13], data_input[:, 14]
        l_ank, r_ank = data_input[:, 15], data_input[:, 16]

    elif J == 18:
        # OpenPose 18点格式 (你的原始代码逻辑)
        # 0:Nose, 1:Neck, 2:R_Sho, 3:R_Elb, 4:R_Wri, 5:L_Sho, 6:L_Elb, 7:L_Wri,
        # 8:R_Hip, 9:R_Kne, 10:R_Ank, 11:L_Hip, 12:L_Kne, 13:L_Ank ...
        print("ℹ️ 检测到 OpenPose 18点格式")

        nose = data_input[:, 0]
        neck = data_input[:, 1]
        r_sho, l_sho = data_input[:, 2], data_input[:, 5]
        r_elb, l_elb = data_input[:, 3], data_input[:, 6]
        r_wri, l_wri = data_input[:, 4], data_input[:, 7]
        r_hip, l_hip = data_input[:, 8], data_input[:, 11]
        r_kne, l_kne = data_input[:, 9], data_input[:, 12]
        r_ank, l_ank = data_input[:, 10], data_input[:, 13]
    else:
        raise ValueError(f"❌ 未知骨架点数: {J}。目前只支持 17(COCO) 或 18(OpenPose)")

    # === 公共计算部分 (生成 H36M) ===
    # MotionBERT H36M 定义:
    # 0:Pelvis, 1:R_Hip, 2:R_Kne, 3:R_Ank, 4:L_Hip, 5:L_Kne, 6:L_Ank,
    # 7:Spine, 8:Neck, 9:Nose, 10:Head, 11:L_Sho, 12:L_Elb, 13:L_Wri,
    # 14:R_Sho, 15:R_Elb, 16:R_Wri

    pelvis = (r_hip + l_hip) / 2.0
    spine = (pelvis + neck) / 2.0
    head = nose + (nose - neck)  # 简单估算头顶

    new_data[:, 0] = pelvis
    new_data[:, 1] = r_hip;
    new_data[:, 2] = r_kne;
    new_data[:, 3] = r_ank
    new_data[:, 4] = l_hip;
    new_data[:, 5] = l_kne;
    new_data[:, 6] = l_ank
    new_data[:, 7] = spine
    new_data[:, 8] = neck
    new_data[:, 9] = nose
    new_data[:, 10] = head
    new_data[:, 11] = l_sho;
    new_data[:, 12] = l_elb;
    new_data[:, 13] = l_wri
    new_data[:, 14] = r_sho;
    new_data[:, 15] = r_elb;
    new_data[:, 16] = r_wri

    # Root Centering (归一化到骨盆为原点)
    new_data -= new_data[:, 0:1, :]

    return new_data


def visualize_skeleton_2d(skeleton, title="Skeleton",sava_path="skeleton.png"):
    plt.figure()
    x = skeleton[:, 0]
    y = skeleton[:, 1]

    plt.scatter(x, y, s=20, c='red')
    for i in range(len(skeleton)):
        plt.text(x[i], y[i], str(i), fontsize=8, color='blue')

    for u, v in SKELETON_EDGES:
        plt.plot([x[u], x[v]], [y[u], y[v]], 'k-')

    plt.title(title)
    # 图像坐标系通常原点在左上角，y向下增大
    # 如果你的数据是图像坐标，加上这一行；如果是世界坐标(米)，可能需要去掉
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.savefig(sava_path)
    plt.close()
    print(f"🖼️ 已保存: {sava_path}")


def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 目录不存在: {DATA_DIR}")
        return
    pkl_files=[]
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".pkl"):
                pkl_files.append(os.path.join(root, f))
    if len(pkl_files) == 0:
        print("❌ 没有找到任何 pkl 文件")
        return

    # 随机取一个文件看，别只看第一个，防止运气好
    import random
    file_path = os.path.join(DATA_DIR, random.choice(pkl_files))

    print(f"📂 加载文件: {file_path}")
    with open(file_path, "rb") as f:
        # CASIA-B pkl 有时候是 dict, 有时候直接是 data
        raw_content = pickle.load(f)

    # 处理可能的 pickle 结构
    if isinstance(raw_content, dict):
        # 很多 gait 库会存成 {'keypoints': ..., 'frame_list': ...}
        # 这里需要你根据实际情况 print(raw_content.keys()) 调试一下
        print(f"ℹ️ Pickle 是字典，Keys: {raw_content.keys()}")
        if 'keypoints' in raw_content:
            raw_data = raw_content['keypoints']
        else:
            # 假设第一个 value 是数据，或者你需要手动看一眼
            raw_data = list(raw_content.values())[0]
    else:
        raw_data = raw_content

    # 确保是 Numpy
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array(raw_data)

    print(f"📊 原始数据形状: {raw_data.shape}")

    try:
        mapped_data = openpose_to_h36m(raw_data)
        print(f"✅ 映射成功! 形状: {mapped_data.shape}")

        # 归一化检查
        y_max = mapped_data[:, :, 1].max()
        print(f"📏 Y轴范围: {mapped_data[:, :, 1].min():.2f} ~ {y_max:.2f}")

        if y_max > 200:
            print("⚠️ 警告: 数据似乎是像素坐标 (e.g., 1080p)。")
            print("💡 建议: 在 dataset 里除以图像高度 (H) 进行归一化，让数值在 -1~1 或 0~1 之间。")
            print("   MotionBERT 虽然对 scale 鲁棒，但数值过大会导致 Transformer 注意力计算溢出。")

        visualize_skeleton_2d(mapped_data[0], "frame0.png", "skeleton.png")

    except Exception as e:
        print(f"❌ 映射失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()