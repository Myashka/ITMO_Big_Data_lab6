import dataclasses
from typing import Optional


@dataclasses.dataclass
class KMeansConfig:
    k: int = dataclasses.field(default=2)
    maxIter: int = dataclasses.field(default=20)
    seed: Optional[int] = dataclasses.field(default=None)


@dataclasses.dataclass
class DataConfig:
    data_path: str = dataclasses.field(default="data/openfood.csv")
    feature_path: str = dataclasses.field(default="configs/features.json")


@dataclasses.dataclass
class SparkConfig:
    app_name: str = dataclasses.field(default="food_cluster")
    deploy_mode: str = dataclasses.field(default="local")
    driver_memory: str = dataclasses.field(default="4g")
    executor_memory: str = dataclasses.field(default="16g")
    executor_cores: int = dataclasses.field(default=1)
    driver_cores: int = dataclasses.field(default=1)

@dataclasses.dataclass
class TrainConfig:
    kmeans: KMeansConfig = dataclasses.field(default_factory=KMeansConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    spark: SparkConfig = dataclasses.field(default_factory=SparkConfig)
    save_to: str = dataclasses.field(default="models/food_cluster")