from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix       |
            |-----------------------|-------------------|
            | PyTorch               | *.pt              |
            | ONNX Runtime          | *.onnx            |

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
    models across various platforms.
    """

    @torch.no_grad()
    def __init__(
            self,
            weights,
            device=torch.device("cpu"),
            fp16=False,
            batch=1,
            verbose=True,
    ):
        """
        Initialize the AutoBackend for inference.

        Args:
            weights (str): Path to the model weights file. Defaults to '.pt'.
            device (torch.device): Device to run the model on. Defaults to CPU.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
            batch (int): Batch-size to assume for inference.
            verbose (bool): Enable verbose logging. Defaults to True.
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
        ) = self._model_type(w)
        fp16 &= pt  # FP16
        model, metadata, task = None, None, None

        # Set device
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if cuda and not any([nn_module, pt, ]):  # GPU dataloader formats
            device = torch.device("cpu")
            cuda = False

        # In-memory PyTorch model
        if nn_module:
            model = weights.to(device)
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True

        # PyTorch
        elif pt:
            from nn.tasks import attempt_load_one_weight
            model, _ = attempt_load_one_weight(weights, is_train=False)
            model = model.half() if fp16 else model.float()
            self.model = model.to(device)

        # Any other format (unsupported)
        else:
            raise TypeError(f"model='{w}' is not a supported model format.")

        # Disable gradients
        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, input_dict):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            input_dict (dict): {
                'pts': torch.FloatTensor(pts),
                'rgb': torch.FloatTensor(rgb),
                'rgb_choose': torch.IntTensor(rgb_choose).long(),
                'translation_label': torch.FloatTensor(target_t),
                'rotation_label': torch.FloatTensor(target_R),
                'tem1_rgb': torch.FloatTensor(tem1_rgb),
                'tem1_choose': torch.IntTensor(tem1_choose).long(),
                'tem1_pts': torch.FloatTensor(tem1_pts),
                'tem2_rgb': torch.FloatTensor(tem2_rgb),
                'tem2_choose': torch.IntTensor(tem2_choose).long(),
                'tem2_pts': torch.FloatTensor(tem2_pts),}


        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """

        if self.fp16:
            for k, v in input_dict.items():
                if torch.is_floating_point(v):
                    input_dict[k] = v.half()  # 将浮点类型转换为FP16

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(input_dict)
        else:
            raise TypeError(f"model format not supported for input_dict inference.")

        if not isinstance(y, dict):
            raise ValueError("Model output is not a dictionary. Please check the model output format.")

        # 确保存在这三个关键输出
        output_dict = {}
        for key in ['pred_R', 'pred_t', 'pred_pose_score']:
            if key in y:
                val = y[key]
                # 若为numpy数组，则转换为tensor
                if isinstance(val, np.ndarray):
                    val = self.from_numpy(val)
                output_dict[key] = val
            else:
                raise KeyError(f"Expected key '{key}' not found in model output.")

        return output_dict

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, input_shapes):
        """
        使用字典输入方式对模型进行热身 (warmup)，通过一次前向传播确保模型初始化和加速。
        Args:
            input_shapes (dict, optional): 指定输入字典中各项的张量形状，用于创建dummy数据进行warmup。
                                           若未提供，将使用默认的假设形状。
                                           格式示例（需根据实际模型输入要求进行修改）:
                                           {
                                               'pts': (1, 1000, 3),
                                               'rgb': (1, 1000, 3),
                                               'rgb_choose': (1, 1000),
                                               'translation_label': (1, 3),
                                               'rotation_label': (1, 3, 3),
                                               'tem1_rgb': (1, 200, 3),
                                               'tem1_choose': (1, 200),
                                               'tem1_pts': (1, 200, 3),
                                               'tem2_rgb': (1, 200, 3),
                                               'tem2_choose': (1, 200),
                                               'tem2_pts': (1, 200, 3),
                                           }
        """
        warmup_types = self.pt, self.nn_module
        if not any(warmup_types):
            return

        if self.device.type == "cpu" and not self.triton:
            return

        if input_shapes is None:
            raise ValueError("input_shapes must be provided for warmup.")

        # 根据 fp16 或 fp32 创建dummy张量
        dtype = torch.half if self.fp16 else torch.float
        input_dict = {}
        for key, shape in input_shapes.items():
            # 简单使用 torch.empty 创建占位张量
            # 逻辑上各输入代表的含义需根据实际模型做对应dummy值生成
            # 此处仅作为warmup占位，不影响实际推理结果
            if 'choose' in key:
                # choose 通常是索引或int类型张量，使用long类型填充
                input_dict[key] = torch.zeros(shape, dtype=torch.long, device=self.device)
            else:
                input_dict[key] = torch.empty(shape, dtype=dtype, device=self.device)

        self.forward(input_dict)

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml, engine, coreml,
        saved_model, pb, tflite, edgetpu, tfjs, ncnn or paddle.

        Args:
            p: path to the model file. Defaults to path/to/model.pt

        Examples:
            >>> model = AutoBackend(weights="path/to/model.onnx")
            >>> model_type = model._model_type()  # returns "onnx"
        """
        from engine.exporter import export_formats
        from utils.checks import check_suffix

        sf = export_formats()["Suffix"]  # export suffixes
        if not isinstance(p, str):
            check_suffix(p, sf)
        name = Path(p).name
        types = [s in name for s in sf]

        return types
