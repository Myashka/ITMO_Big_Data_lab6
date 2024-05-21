import pyrallis

from src import configs, train


@pyrallis.wrap()
def main(config: configs.TrainConfig):
    train.run(config)


if __name__ == "__main__":
    main()
