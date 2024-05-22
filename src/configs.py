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
class DatabaseConfig:
    server: str = dataclasses.field(default="mssql-server")
    database: str = dataclasses.field(default="FoodData")
    username: str = dataclasses.field(default="sa")
    password: str = dataclasses.field(default="yourStrong(!)Password")
    driver: str = dataclasses.field(default="com.microsoft.sqlserver.jdbc.SQLServerDriver")
    url: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.url = (f"jdbc:sqlserver://{self.server};databaseName={self.database};"
                    f"user={self.username};password={self.password};"
                    f"encrypt=false;trustServerCertificate=true")


@dataclasses.dataclass
class TrainConfig:
    kmeans: KMeansConfig = dataclasses.field(default_factory=KMeansConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    spark: SparkConfig = dataclasses.field(default_factory=SparkConfig)
    db: DatabaseConfig = dataclasses.field(default_factory=DatabaseConfig)
    save_to: str = dataclasses.field(default="models/food_cluster")