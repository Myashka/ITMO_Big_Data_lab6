import json

import pyspark.sql
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions

from src.configs import DataConfig

FEATURES_COLUMN = "scaled_feature"


class Preprocessor:
    def __init__(self, data_config: DataConfig):
        with open(data_config.feature_path, "r") as features_file:
            self.features = json.load(features_file)

        self.data_config = data_config

    # def load_data(self, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    #     df = spark.read.csv(self.data_config.data_path, header=True, inferSchema=True)

    #     return df

    def preprocess(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:

        id_columns = self.features["id"]
        feature_numeric = self.features["numeric"]
        numeric_columns = [
            functions.col(c).cast("float").alias(c) for c in feature_numeric
        ]
        cat_columns = self.features["categorical"]

        all_columns = id_columns + numeric_columns + cat_columns
        df_with_selected_columns = df.select(*all_columns)

        vec_assembler = VectorAssembler(
            inputCols=feature_numeric, outputCol="features"
        )
        df_with_features = vec_assembler.transform(df_with_selected_columns)

        scaler = StandardScaler(inputCol="features", outputCol=FEATURES_COLUMN)
        scaler_model = scaler.fit(df_with_features)
        df_scaled_features = scaler_model.transform(df_with_features)
        return df_scaled_features
