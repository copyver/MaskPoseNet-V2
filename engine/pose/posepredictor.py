from engine import BasePredictor


class PosePredictor(BasePredictor):
    def __init__(self, config, model, device):
        super(PosePredictor, self).__init__(config, model, device)
        pass