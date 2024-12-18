import torch
from models.pose.posemodel import PoseModel
from easydict import EasyDict as edict
from engine.model import Model

# 加载模型文件
# checkpoint = torch.load("middle_log/20241216_Test_PoseTrain/checkpoints/best.pt", map_location="cpu")
#
# # # 查看 checkpoint 中的内容
# print(checkpoint.keys())
# print(checkpoint['train_args'])
# # model = PoseModel(edict(checkpoint['train_args']).POSE_MODEL)
# # model.load_state_dict(checkpoint["model"])
#
# model = checkpoint['model']
# # 转换为 FP16
# model.half()
# model.eval()

model = Model(
    model="middle_log/1217_train/checkpoints/last.pt",
    task='pose',
    verbose=False
)
source = {
    "image": "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
             "datasets_handle_1000test/images/color_ims/image_000000.png",
    "depth_image": "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
                   "datasets_handle_1000test/images/depth_ims/image_000000.png",
    "seg_mask": "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
                "datasets_handle_1000test/images/modal_masks/image_000000/channel_012.png",
    "obj": [1],
    "camera_k": [
        [1056.3635127452394, 0.0, 640.3173115306723, ],
        [0.0, 1056.3635127452394, 479.3086393011287, ],
        [0.0, 0.0, 1.0],
    ],
}
model.predict(source)
