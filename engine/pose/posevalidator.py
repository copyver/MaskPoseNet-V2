from engine import BaseValidator


class PoseValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, cfg=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, cfg, _callbacks)
