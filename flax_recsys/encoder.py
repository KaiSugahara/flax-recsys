from typing import Self

import jax
import jax.numpy as jnp
import polars as pl


class ColumnEncoder:
    encoding_map: dict[str, dict] = {}
    cardinality_map: dict[str, int] = {}
    field_sizes: list[int] = []

    def __init__(
        self,
        column_setting: dict,
    ):
        self.column_setting = {
            "user_id": None,
            "item_id": None,
            "rating": None,
            "timestamp": None,
            "one_hot": [],
            "multi_hot": [],
            "numeric": [],
        } | column_setting.copy()

    def fit(
        self,
        dataset_df: pl.DataFrame,
    ) -> Self:
        # Re-Initialize
        self.encoding_map = {}
        self.cardinality_map = {}
        self.field_sizes = []

        # Build encoding_map & cardinality_map & field_sizes
        for column_name in (
            [
                self.column_setting["user_id"],
                self.column_setting["item_id"],
            ]
            + self.column_setting["one_hot"]
            + self.column_setting["multi_hot"]
        ):
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
            self.field_sizes += [len(self.encoding_map[column_name]) + 1]

        for column_name in self.column_setting["numeric"]:
            self.field_sizes += [1]

        return self

    def transform(self, dataset_df: pl.DataFrame) -> tuple[jax.Array, jax.Array]:
        arr = []

        # One-hot Type Column
        for column_name in [
            self.column_setting["user_id"],
            self.column_setting["item_id"],
        ] + self.column_setting["one_hot"]:
            arr.append(
                jax.nn.one_hot(
                    dataset_df.get_column(column_name)
                    .replace_strict(self.encoding_map[column_name], default=0)
                    .to_numpy(),
                    self.cardinality_map[column_name],
                )
            )

        # Multi-hot Type Column
        for column_name in self.column_setting["multi_hot"]:
            arr.append(
                jnp.vstack(
                    [
                        jax.nn.one_hot(
                            row.to_numpy(), self.cardinality_map[column_name]
                        ).sum(axis=0)
                        for row in dataset_df.get_column(column_name).list.eval(
                            pl.element().replace_strict(
                                self.encoding_map[column_name], default=0
                            )
                        )
                    ]
                )
            )

        # Numeric Type Column
        if len(self.column_setting["numeric"]) > 0:
            arr.append(
                jax.device_put(
                    dataset_df.select(self.column_setting["numeric"]).to_numpy()
                )
            )

        return jnp.hstack(arr), jax.device_put(dataset_df.select("rating").to_numpy())

    @property
    def dimension(self) -> int:
        dim = 0

        # One-hot Type Column
        for column_name in [
            self.column_setting["user_id"],
            self.column_setting["item_id"],
        ] + self.column_setting["one_hot"]:
            dim += self.cardinality_map[column_name]

        # Multi-hot Type Column
        for column_name in self.column_setting["multi_hot"]:
            dim += self.cardinality_map[column_name]

        # Numeric Type Column
        for column_name in self.column_setting["numeric"]:
            dim += 1

        return dim
