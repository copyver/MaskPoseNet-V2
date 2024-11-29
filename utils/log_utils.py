import numpy as np

class BaseWriter:
    """ base class  to write train or test logs"""
    def write(self, data, iter):
        """
        Args
            data(dict): log data
            iter(int): iter number
        """
        raise NotImplementedError

    def close(self):
        pass

class TensorBoardWriter(BaseWriter):
    def __init__(self, log_dir: (str, Path), window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        log_dir = str(log_dir)
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)

    def write(self, data:dict, iter:int):
        for key, value in data.items():
            # image data
            if key is "images":
                assert isinstance(value, np.ndarray), "tensorboard images not support format {}".format(type(value))
                self._writer.add_images(data["image_names"], value, iter)
            # loss map data ....
            else:
                if isinstance(value, str) and key is "image_names":
                    continue

                assert isinstance(value, float), "tensorboard scalar not support format {}".format(type(value))
                self._writer.add_scalar(key, value, iter)

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()