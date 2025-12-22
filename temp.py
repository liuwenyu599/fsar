import pickle
import numpy as np

pkl_path = "/datasets/CASIA-B/pose/CASIA-B_HRNet/059/nm-02/144/144.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("type:", type(data))
print("shape:", data.shape)
print("dtype:", data.dtype)

# 打印前一帧看看
print("first frame sample:\n", data[0])
