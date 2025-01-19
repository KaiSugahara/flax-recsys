import math
from typing import Self

import jax
from flax import nnx
from flax_trainer.loader import BaseLoader


class GeneralRatingLoader(BaseLoader):
    def __init__(
        self,
        categorical_X: jax.Array,
        numerical_X: jax.Array,
        y: jax.Array,
        batch_size: int,
        rngs: nnx.Rngs,
    ):
        self.categorical_X = categorical_X
        self.numerical_X = numerical_X
        self.y = y
        self.batch_size = batch_size
        self.rngs = rngs

    def __iter__(self) -> Self:
        """Prepares for batch iteration"""

        # Num. of data
        self.data_size = self.categorical_X.shape[0]

        # Num. of batch
        self.batch_num = math.ceil(self.data_size / self.batch_size)

        # Shuffle rows of data
        self.shuffled_indices = jax.random.permutation(self.rngs(), self.data_size)
        self.shuffled_categorical_X, self.shuffled_numerical_X, self.shuffled_y = (
            self.categorical_X[self.shuffled_indices],
            self.numerical_X[self.shuffled_indices],
            self.y[self.shuffled_indices],
        )

        # Initialize batch index
        self.batch_index = 0

        return self

    def __len__(self) -> int:
        """Returns the number of batches

        Returns:
            int: The number of batches
        """

        return self.batch_num

    def __next__(self) -> tuple[tuple[jax.Array, ...], jax.Array]:
        """Returns data from the current batch

        Returns:
            jax.Array: The input data.
            jax.Array: The target data.
        """

        if self.batch_index >= self.batch_num:
            raise StopIteration()

        else:
            # Extract the {batch_index}-th mini-batch
            start_index = self.batch_size * self.batch_index
            slice_size = min(self.batch_size, (self.data_size - start_index))
            categorical_X, numerical_X, y = (
                jax.lax.dynamic_slice_in_dim(
                    self.shuffled_categorical_X, start_index, slice_size
                ),
                jax.lax.dynamic_slice_in_dim(
                    self.shuffled_numerical_X, start_index, slice_size
                ),
                jax.lax.dynamic_slice_in_dim(self.shuffled_y, start_index, slice_size),
            )

            # Update batch index
            self.batch_index += 1

            return (categorical_X, numerical_X), y
