# challenge/model.py

from typing import List, Tuple, Union

import pandas as pd
from xgboost import XGBClassifier


class DelayModel:
    """Flight delay prediction model."""

    TOP_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    RAW_FEATURE_COLUMNS = ["OPERA", "TIPOVUELO", "MES"]

    def __init__(self):
        self._model = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        The selected model only uses the Top 10 features from the notebook,
        derived from OPERA, TIPOVUELO and MES.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): target column name. If provided,
                the target is returned together with the features.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        df = data.copy()

        features = self._build_features(df)

        if target_column is None:
            return features

        target = self._build_target(df, target_column)
        return features, target

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
    ) -> None:
        """
        Fit the XGBoost model using class balancing.

        Args:
            features (pd.DataFrame): preprocessed features.
            target (pd.DataFrame): target.
        """
        y = target.iloc[:, 0].astype(int)

        negative_class = (y == 0).sum()
        positive_class = (y == 1).sum()

        scale_pos_weight = (
            negative_class / positive_class if positive_class > 0 else 1.0
        )

        self._model = XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
        )

        self._model.fit(features, y)

    def predict(
        self,
        features: pd.DataFrame,
    ) -> List[int]:
        """
        Predict delay labels.

        Args:
            features (pd.DataFrame): preprocessed features.

        Returns:
            List[int]: predicted targets.
        """
        if self._model is None:
            return [0] * features.shape[0]

        predictions = self._model.predict(features)
        return predictions.astype(int).tolist()

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build the Top 10 model features from raw input columns.
        """
        self._validate_columns(data, self.RAW_FEATURE_COLUMNS)

        df = data[self.RAW_FEATURE_COLUMNS].copy()
        df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(-1).astype(int)

        encoded_features = pd.concat(
            [
                pd.get_dummies(df["OPERA"], prefix="OPERA"),
                pd.get_dummies(df["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(df["MES"], prefix="MES"),
            ],
            axis=1,
        )

        return encoded_features.reindex(columns=self.TOP_FEATURES, fill_value=0).astype(
            int
        )

    def _build_target(
        self,
        data: pd.DataFrame,
        target_column: str,
    ) -> pd.DataFrame:
        """
        Build the target column.

        If the target already exists, it is reused. Otherwise, it is created
        from the difference between Fecha-O and Fecha-I.
        """
        if target_column in data.columns:
            target = data[target_column].astype(int)
            return pd.DataFrame({target_column: target})

        self._validate_columns(data, ["Fecha-I", "Fecha-O"])

        scheduled_date = pd.to_datetime(data["Fecha-I"], errors="coerce")
        operation_date = pd.to_datetime(data["Fecha-O"], errors="coerce")

        min_diff = (operation_date - scheduled_date).dt.total_seconds() / 60
        target = (min_diff > 15).astype(int)

        return pd.DataFrame({target_column: target})

    @staticmethod
    def _validate_columns(
        data: pd.DataFrame,
        required_columns: List[str],
    ) -> None:
        """
        Validate that all required columns are present.
        """
        missing_columns = [
            column for column in required_columns if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
