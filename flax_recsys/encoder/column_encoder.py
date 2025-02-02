from typing import Self

import jax
import jax.numpy as jnp
import polars as pl
from chex import dataclass


@dataclass
class ColumnSetting:
    user_id: None | str
    item_id: None | str
    rating: None | str
    timestamp: None | str
    one_hot: list[str]
    multi_hot: list[str]
    numerical: list[str]


class ColumnEncoder:
    def __init__(
        self,
        *,
        user_id: None | str = None,
        item_id: None | str = None,
        rating: None | str = None,
        timestamp: None | str = None,
        one_hot: list[str] = [],
        multi_hot: list[str] = [],
        numerical: list[str] = [],
    ):
        # Set column setting
        self.column_setting = ColumnSetting(
            user_id=user_id,
            item_id=item_id,
            rating=rating,
            timestamp=timestamp,
            one_hot=one_hot,
            multi_hot=multi_hot,
            numerical=numerical,
        )

        # Initialize
        self.encoding_map: dict[str, dict] = {}
        self.cardinality_map: dict[str, int] = {}

    def fit(
        self,
        dataset_df: pl.DataFrame,
    ) -> Self:
        # Re-Initialize
        self.encoding_map = {}
        self.cardinality_map = {}

        # Build encoding_map & cardinality_map for categorical columns
        for column_name in self.categorical_columns:
            self.encoding_map[column_name] = dict(
                zip(
                    (
                        uni_list := dataset_df.get_column(column_name)
                        .explode()
                        .unique()
                        .sort()
                    ),
                    range(1, len(uni_list) + 1),
                )
            )
            self.cardinality_map[column_name] = len(self.encoding_map[column_name]) + 1

        # Build encoding_map & cardinality_map for multihot columns
        for column_name in self.column_setting.multi_hot:
            raise NotImplementedError()

        return self

    def transform(
        self, dataset_df: pl.DataFrame
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # One-hot Type Column Matrix
        categorical_X_df = dataset_df.select(
            pl.col(column_name).replace_strict(
                self.encoding_map[column_name], default=0
            )
            for column_name in self.categorical_columns
        )
        categorical_X = (
            jax.device_put(categorical_X_df.to_numpy())
            if categorical_X_df.shape[1] > 0
            else jnp.empty((dataset_df.height, 0))
        )

        # Multi-hot Type Column (Not yet implemented)
        if len(self.column_setting.multi_hot) > 0:
            raise NotImplementedError()

        # Numeric Type Column
        numerical_X_df = dataset_df.select(self.numerical_columns).cast(pl.Float32)
        numerical_X = (
            jax.device_put(numerical_X_df.to_numpy())
            if numerical_X_df.shape[1] > 0
            else jnp.empty((dataset_df.height, 0))
        )

        # Target Column Matrix
        y_df = dataset_df.select(self.target_columns).cast(pl.Float32)
        y = (
            jax.device_put(y_df.to_numpy())
            if y_df.shape[1] > 0
            else jnp.empty((dataset_df.height, 0))
        )

        return categorical_X, numerical_X, y

    def fit_transform(self, dataset_df: pl.DataFrame):
        return self.fit(dataset_df=dataset_df).transform(dataset_df=dataset_df)

    @property
    def categorical_columns(self) -> list[str]:
        return (
            (
                [self.column_setting.user_id]
                if self.column_setting.user_id is not None
                else []
            )
            + (
                [self.column_setting.item_id]
                if self.column_setting.item_id is not None
                else []
            )
            + self.column_setting.one_hot
        )

    @property
    def numerical_columns(self) -> list[str]:
        return self.column_setting.numerical

    @property
    def target_columns(self) -> list[str]:
        return (
            [self.column_setting.rating]
            if self.column_setting.rating is not None
            else []
        )

    @property
    def categorical_column_num(self) -> int:
        return len(self.categorical_columns)

    @property
    def numerical_column_num(self) -> int:
        return len(self.numerical_columns)
