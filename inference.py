import torch
from models.pose.posemodel import PoseModel
from easydict import EasyDict as edict
# 加载模型文件
checkpoint = torch.load("middle_log/20241216_Test_PoseTrain/checkpoints/last.pt", map_location="cpu")

# # 查看 checkpoint 中的内容
# print(checkpoint.keys())
# print(checkpoint['train_args'])
# model = PoseModel(edict(checkpoint['train_args']).POSE_MODEL)
# model.load_state_dict(checkpoint["model"])

model = checkpoint['model']
# 转换为 FP16
model.half()
model.eval()