import torch
from torch.utils.cpp_extension import CUDA_HOME


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print(torch.cuda.is_available())  # 输出 True 表示 CUDA 可用
    print(torch.version.cuda)  # 查看 PyTorch 使用的 CUDA 版本
    print("CUDA_HOME:", CUDA_HOME)


