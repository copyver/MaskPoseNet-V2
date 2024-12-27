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
        model="middle_log/1223_train/checkpoints/last.pt",
        task='pose',
        verbose=False
    )
    # source = {
    #     "image": "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #              "datasets_handle_1000test/images/color_ims/image_000000.png",
    #     "depth_image": "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                    "datasets_handle_1000test/images/depth_ims/image_000000.png",
    #     "seg_mask": ["/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_000.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_001.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_002.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_003.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_004.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_005.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_006.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_007.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_008.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_009.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_010.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_011.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_012.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_013.png",
    #                  "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/"
    #                  "datasets_handle_1000test/images/modal_masks/image_000000/channel_014.png",
    #                  ],
    #     "obj": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
    #     "camera_k": np.array([
    #         [1056.3635127452394, 0.0, 640.3173115306723, ],
    #         [0.0, 1056.3635127452394, 479.3086393011287, ],
    #         [0.0, 0.0, 1.0],
    #     ], dtype=np.float64),
    # }
    source = {
        "image": "/home/yhlever/CLionProjects/ROBOT_GRASP_TORCH/results/color.png",
        "depth_image": "/home/yhlever/CLionProjects/ROBOT_GRASP_TORCH/results/depth.png",
        "seg_mask": ["/home/yhlever/CLionProjects/ROBOT_GRASP_TORCH/results/mask_0.png",
                     ],
        "obj": [1],
        "camera_k": np.array([
            [1062.67, 0.0, 646.17, ],
            [0.0, 1062.67, 474.24, ],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),
     }
    result = model.predict(source, override="cfg/base.yaml")
    print(result)
    print("Successful Inference")
