# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Export a  PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | .pt
TorchScript             | `torchscript`             | .torchscript
ONNX                    | `onnx`                    | .onnx
TensorRT                | `engine`                  | yolo11n.engine


Python:
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolo11n.pt format=onnx



TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolo11n_web_model public/yolo11n_web_model
    $ npm start
"""

import gc
import json
import os
import shutil
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from utils import get_default_args
import numpy as np
import torch
from utils.ops import Profile
from loguru import logger
from utils.files import file_size
from utils.torch_utils import smart_inference_mode, select_device, get_latest_opset
from models.pose import get_default_tensor, get_default_input_shape
from utils import __version__, colorstr, LINUX
from utils.checks import check_version
from data.dataloader.build import build_dataloader
from data.dataset.posenet_dataset import PoseNetDataset


def export_formats():
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["TensorRT", "engine", ".engine", False, True],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU"], zip(*x)))


def try_export(inner_func):
    """YOLO export decorator, i.e. @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            logger.info(f"{prefix} export success  {dt.t:.1f}s, saved as '{f}' ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            logger.error(f"{prefix} export failure {dt.t:.1f}s: {e}")
            raise e

    return outer_func


class Exporter:
    """
    A class for exporting a model.

    Attributes:
        cfg (SimpleNamespace): Configuration for the exporter.
        callbacks (list, optional): List of callback functions. Defaults to None.
    """

    def __init__(self, cfg=None, _callbacks=None):
        """
        Initializes the Exporter class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            _callbacks (dict, optional): Dictionary of callback functions. Defaults to None.
        """
        self.cfg = cfg.EXPORT
        self.data_cfg = cfg.TEST_DATA
        self.callbacks = _callbacks
        self.input_dict = get_default_input_shape(cfg)

    @smart_inference_mode()
    def __call__(self, model=None) -> str:
        """Returns list of exported files/dirs after running callbacks."""
        self.run_callbacks("on_export_start")
        t = time.time()
        fmt = self.cfg.FORMAT.lower()  # to lowercase
        if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
            fmt = "engine"

        fmts = tuple(export_formats()["Argument"][1:])  # available export formats
        if fmt not in fmts:
            logger.warning(f"WARNING  Invalid export format='{fmt}'")

        # Device
        if fmt == "engine" and self.cfg.DEVICE is None:
            logger.warning("WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0")
            self.cfg.DEVICE = "0"

        flags = [x == fmt for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")
        jit, onnx, engine = flags

        self.device = select_device("cpu" if self.cfg.DEVICE is None else self.cfg.DEVICE)

        # Checks
        # if not hasattr(model, "names"):
        #     model.names = default_class_names()
        # model.names = check_class_names(model.names)
        if self.cfg.HALF and self.cfg.INT8:
            logger.warning("WARNING half=True and int8=True are mutually exclusive, setting half=False.")
            self.cfg.HALF = False
        if self.cfg.HALF and onnx and self.device.type == "cpu":
            logger.warning("WARNING half=True only compatible with GPU export, i.e. use device=0")
            self.cfg.HALF = False
            assert not self.cfg.DYNAMIC, "half=True not compatible with dynamic=True, i.e. use only one."

        if self.cfg.INT8 and engine:
            self.cfg.DYNAMIC = True  # enforce dynamic to export TensorRT INT8
        if self.cfg.OPTIMIZE:
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"

        if self.cfg.INT8 and not self.cfg.data:
            pass  # Todo:

        # Input
        input_tensor = get_default_tensor(self.input_dict, self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        # model = model.fuse()

        y = None
        for _ in range(2):
            y = model(input_tensor)  # dry runs
        if self.cfg.HALF and onnx and self.device.type != "cpu":
            if isinstance(input_tensor, dict):
                # 如果 input_tensor 是字典，对字典的每个值调用 .half()
                input_tensor = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in input_tensor.items()}
            else:
                # 否则直接对整个 input_tensor 调用 .half()
                input_tensor = input_tensor.half()
            model = model.half()

        # Filter warnings
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.input_tensor = input_tensor
        self.model = model
        self.file = file
        self.input_shape = (
            {k: tuple(v.shape) if isinstance(v, torch.Tensor) else None for k, v in input_tensor.items()}
            if isinstance(input_tensor, dict)
            else tuple(input_tensor.shape if isinstance(input_tensor, torch.Tensor) else [])
        )
        self.output_shape = (
            {k: tuple(v.shape) if isinstance(v, torch.Tensor) else None for k, v in y.items()}
            if isinstance(y, dict)
            else tuple(y.shape if isinstance(y, torch.Tensor) else [])
        )
        description = f'{self.model.names}'
        self.metadata = {
            "description": description,
            "author": "yhlever",
            "date": datetime.now().isoformat(),
            "version": __version__,
        }  # model metadata

        logger.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {self.input_shape} and "
            f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)'
        )

        # Exports
        f = [""] * len(fmts)  # exported filenames
        if jit:
            f[0], _ = self.export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1], _ = self.export_engine()
        if onnx:  # ONNX
            f[2], _ = self.export_onnx()

        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            f = str(Path(f[-1]))
            q = "int8" if self.cfg.int8 else "half" if self.cfg.half else ""  # quantization
            logger.info(
                f'\nExport complete ({time.time() - t:.1f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f'\nPredict:         yolo predict task={model.task} model={f} {q} '
                f'\nValidate:        yolo val task={model.task} model={f} {q} '
                f'\nVisualize:       https://netron.app'
            )

        self.run_callbacks("on_export_end")
        return f  # return list of exported files/dirs

    def get_int8_calibration_dataloader(self, prefix=""):
        """Build and return a dataloader suitable for calibration of INT8 models."""
        logger.info(f"{prefix} collecting INT8 calibration images from 'data={self.cfg.data}'")
        batch = self.data_cfg.BATCH_SIZE * (2 if self.cfg.FORMAT == "engine" else 1)
        dataset = PoseNetDataset(self.data_cfg, is_train=False)

        return build_dataloader(dataset, batch=batch, workers=0)  # required for batch loading

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """YOLO TorchScript model export."""
        logger.info(f"\n{prefix} starting export with torch {torch.__version__}...")
        f = self.file.with_suffix(".torchscript")

        # ts = torch.jit.trace(self.model, self.input_tensor, strict=False)
        # extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
        # if self.cfg.OPTIMIZE:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        #     logger.info(f"{prefix} optimizing for mobile...")
        #     from torch.utils.mobile_optimizer import optimize_for_mobile
        #
        #     optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        # else:
        #     ts.save(str(f), _extra_files=extra_files)
        ts = torch.jit.script(self.model, example_inputs=self.input_tensor)
        extra_files = {"config.txt": json.dumps(self.metadata)}
        ts.save(str(f), _extra_files=extra_files)
        return f, None

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """YOLO ONNX export."""
        import onnx  # noqa
        opset_version = get_latest_opset()
        logger.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")
        f = str(self.file.with_suffix(".onnx"))

        output_names = ["endpoints"]
        print(self.input_tensor)
        torch.onnx.export(
            self.model,  # dynamic=True only compatible with cpu
            args=self.input_tensor,
            f=f,
            verbose=False,
            opset_version=opset_version,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=["input_dict"],
            output_names=output_names,
            dynamic_axes=None,
        )

        # Checks
        model_onnx = onnx.load(f)  # load onnx model

        # Simplify
        if self.cfg.SIMPLIFY:
            try:
                import onnxslim

                logger.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)

            except Exception as e:
                logger.warning(f"{prefix} simplifier failure: {e}")

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        onnx.save(model_onnx, f)
        return f, model_onnx


    @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):
        """YOLO TensorRT export https://developer.nvidia.com/tensorrt."""
        assert self.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            raise ImportError("tensorrt is not installed")
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        # Setup and checks
        logger.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        _logger = trt.logger(trt.logger.INFO)
        if self.cfg.verbose:
            _logger.min_severity = trt.logger.Severity.VERBOSE

        # Engine builder
        builder = trt.Builder(_logger)
        config = builder.create_builder_config()
        workspace = int(self.cfg.WORKSPACE * (1 << 30))
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        half = builder.platform_has_fast_fp16 and self.cfg.half
        int8 = builder.platform_has_fast_int8 and self.cfg.int8

        # Read ONNX file
        parser = trt.OnnxParser(network, _logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

        # Network inputs
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            logger.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            logger.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if self.cfg.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                logger.warning(f"{prefix} WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
            profile = builder.create_optimization_profile()
            min_shape = (1, shape[1], 32, 32)  # minimum input shape
            max_shape = (*shape[:2], *(int(max(1, self.cfg.workspace) * d) for d in shape[2:]))  # max input shape
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
            config.add_optimization_profile(profile)

        logger.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_calibration_profile(profile)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            class EngineCalibrator(trt.IInt8Calibrator):
                def __init__(
                    self,
                    dataset,  # ultralytics.data.build.InfiniteDataLoader
                    batch: int,
                    cache: str = "",
                ) -> None:
                    trt.IInt8Calibrator.__init__(self)
                    self.dataset = dataset
                    self.data_iter = iter(dataset)
                    self.algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                    self.batch = batch
                    self.cache = Path(cache)

                def get_algorithm(self) -> trt.CalibrationAlgoType:
                    """Get the calibration algorithm to use."""
                    return self.algo

                def get_batch_size(self) -> int:
                    """Get the batch size to use for calibration."""
                    return self.batch or 1

                def get_batch(self, names) -> list:
                    """Get the next batch to use for calibration, as a list of device memory pointers."""
                    try:
                        im0s = next(self.data_iter)["img"] / 255.0
                        im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                        return [int(im0s.data_ptr())]
                    except StopIteration:
                        # Return [] or None, signal to TensorRT there is no calibration data remaining
                        return None

                def read_calibration_cache(self) -> bytes:
                    """Use existing cache instead of calibrating again, otherwise, implicitly return None."""
                    if self.cache.exists() and self.cache.suffix == ".cache":
                        return self.cache.read_bytes()

                def write_calibration_cache(self, cache) -> None:
                    """Write calibration cache to disk."""
                    _ = self.cache.write_bytes(cache)

            # Load dataset w/ builder (for batching) and calibrate
            config.int8_calibrator = EngineCalibrator(
                dataset=self.get_int8_calibration_dataloader(prefix),
                batch=2 * self.cfg.batch,  # TensorRT INT8 calibration should use 2x batch size
                cache=str(self.file.with_suffix(".cache")),
            )

        elif half:
            config.set_flag(trt.BuilderFlag.FP16)

        # Free CUDA memory
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        # Write file
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            # Metadata
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
            # Model
            t.write(engine if is_trt10 else engine.serialize())

        return f, None

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

