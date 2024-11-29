from engine import BaseTrainer

class PoseTrainer(BaseTrainer):
    def __init__(self, cfg, model):
        super(PoseTrainer, self).__init__(cfg, model)


    def get_dataset(self):
        train_dataset = PoseNetDataset(self.cfg, is_train=True)
        test_dataset = PoseNetDataset(self.cfg, is_train=False)
        return train_dataset, test_dataset