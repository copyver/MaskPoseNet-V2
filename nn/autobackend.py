from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from collections import OrderedDict, namedtuple
import json


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
            weights (str): Path to the model weights file. Defaults to 'yolov8n.pt'.
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
            jit,
            onnx,
            engine,
            triton,
        ) = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or nn_module or triton  # FP16
        model, metadata, task = None, None, None

        # Set device
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx]):  # GPU dataloader formats
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
            from nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=False
            )
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # ONNX Runtime
        elif onnx:
            logger.info(f"Loading {w} for ONNX Runtime inference...")
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            if not cuda and "CUDAExecutionProvider" in providers:
                providers.remove("CUDAExecutionProvider")
            elif cuda and "CUDAExecutionProvider" not in providers:
                logger.warning("WARNING ⚠️ Failed to start ONNX Runtime session with CUDA. Falling back to CPU...")
                device = torch.device("cpu")
                cuda = False
            logger.info(f"Preferring ONNX Runtime {providers[0]}")
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
            dynamic = isinstance(session.get_outputs()[0].shape[0], str)
            if not dynamic:
                io = session.io_binding()
                bindings = []
                for output in session.get_outputs():
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if fp16 else torch.float32).to(device)
                    io.bind_output(
                        name=output.name,
                        device_type=device.type,
                        device_id=device.index if cuda else 0,
                        element_type=np.float16 if fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    bindings.append(y_tensor)

                    # TensorRT
        elif engine:
            logger.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download


            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # Read file
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                except UnicodeDecodeError:
                    f.seek(0)  # engine file may lack embedded Ultralytics metadata
                model = runtime.deserialize_cuda_engine(f.read())  # read engine

            # Model context
            try:
                context = model.create_execution_context()
            except Exception as e:  # model is None
                logger.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
                raise e

            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size

        # Any other format (unsupported)
        else:
            raise TypeError(f"model='{w}' is not a supported model format.")

        # Disable gradients
        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, input_dict, visualize=False, embed=None):
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
            visualize (bool): whether to visualize the output predictions, defaults to False
            embed (list, optional): A list of feature vectors/embeddings to return.

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

        # ONNX Runtime
        elif self.onnx:
            input_feed = {}
            for inp in self.session.get_inputs():
                inp_name = inp.name
                val = input_dict[inp_name]
                val_cpu = val.detach().cpu().numpy()
                input_feed[inp_name] = val_cpu

            if self.dynamic:
                y = self.session.run(self.output_names, input_feed)
                # 动态输出可能是列表或其它格式，需要转换为字典形式
                # 此处需根据您的ONNX模型实际输出对应关系进行映射
                # 假设您的ONNX模型输出与 pred_R, pred_t, pred_pose_score 对应:
                # outputs = ['pred_R', 'pred_t', 'pred_pose_score']
                # y 将是一个列表，对应上述三个输出名顺序
                # 您需要根据实际情况修改此处的映射逻辑
                y_dict = {}
                output_names = [o.name for o in self.session.get_outputs()]
                for name, val in zip(output_names, y):
                    y_dict[name] = val
                y = y_dict
            else:
                # 静态shape情况，需要IO Binding
                self.io.clear_binding_inputs()
                for inp in self.session.get_inputs():
                    inp_name = inp.name
                    val = input_dict[inp_name]
                    val = val.to(self.device) if self.cuda else val
                    self.io.bind_input(
                        name=inp_name,
                        device_type=val.device.type,
                        device_id=val.device.index if val.device.type == "cuda" else 0,
                        element_type=np.float16 if self.fp16 else np.float32,
                        shape=tuple(val.shape),
                        buffer_ptr=val.data_ptr(),
                    )

                self.session.run_with_iobinding(self.io)
                # self.bindings是已绑定的输出列表，与ONNX输出对应，需要映射回字典
                y = {}
                output_tensors = self.bindings
                output_names = [o.name for o in self.session.get_outputs()]
                for name, out in zip(output_names, output_tensors):
                    # out 是 torch.Tensor 而非 numpy 数组，因为我们使用了 IO binding
                    y[name] = out

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
        warmup_types = self.pt, self.onnx, self.nn_module
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

        triton = False
        if any(types):
            triton = False
        return types + [triton]
