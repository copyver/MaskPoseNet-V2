import numpy as np

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


if __name__ == "__main__":
    model = Model(
        model="middle_log/0112-indus12000-train-b/checkpoints/best.pt",
        task='pose',
        verbose=False,
        is_train=False,
        device="cuda",
    )

    source = {
        "image": "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v"
                 "/val/images/color_ims/image_000512.png",
        "depth_image": "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v"
                 "/val/images/depth_ims/image_000512.png",
        "seg_mask": [
                     "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v/val/"
                     "images/modal_masks/image_000512/channel_001.png",
                     "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v/val/"
                     "images/modal_masks/image_000512/channel_002.png",
                     "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v/val/"
                     "images/modal_masks/image_000512/channel_003.png",
                     "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v/val/"
                     "images/modal_masks/image_000512/channel_004.png",
                     "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v/val/"
                     "images/modal_masks/image_000512/channel_005.png",
                     "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v/val/"
                     "images/modal_masks/image_000512/channel_006.png",
                     "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v/val/"
                     "images/modal_masks/image_000512/channel_007.png",
                     "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/indus-12000t-1200v/val/"
                     "images/modal_masks/image_000512/channel_008.png",
                     ],
        "cls_ids": [5, 5, 5, 5, 6, 1, 6, 1],
        "camera_k": np.array([
            [1039.5527733689619, 0.0, 639.3049718803047,],
            [0.0, 1039.5527733689619, 479.5612565995002,],
            [0.0, 0.0, 1.0,],
        ], dtype=np.float64),
    }
    result = model.predict(source)
    print("Successful Inference")
