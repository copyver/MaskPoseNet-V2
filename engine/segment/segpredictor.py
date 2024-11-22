from engine import BasePredictor


class SegPredictor(BasePredictor):
    def __init__(self, config, model, device, logger):
        super().__init__(config, model, device, logger)
        pass