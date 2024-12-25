from engine.model import Model


if __name__ == '__main__':
    model = Model(
        model="middle_log/1223_train/checkpoints/last.pt",
        task='pose',
        verbose=False
    )
    model.export(override="cfg/base.yaml")