from engine.model import Model
from loguru import logger


def main():
    model = Model(
        model='./cfg/base.yaml',
        task='pose',
        verbose=False
    )
    model.train()


if __name__ == "__main__":
    main()
