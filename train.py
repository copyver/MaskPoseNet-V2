from engine.model import Model


def main():
    model = Model(
        model='cfg/tless.yaml',
        task='pose',
        verbose=False,
        is_train=True
    )

    model.train()


if __name__ == "__main__":
    main()
