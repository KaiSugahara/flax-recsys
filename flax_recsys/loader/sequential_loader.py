import random
from typing import Generic, Self, TypeVar

import jax
import numpy as np
from flax import nnx
from flax_trainer.loader import BaseLoader
from tqdm.auto import tqdm

from flax_recsys.encoder.sequential_encoder import SequentialEncoder

T = TypeVar("T", str, int)


class SequentialLoader(BaseLoader, Generic[T]):
    def __init__(
        self,
        sequences: list[list[T]],
        encoder: SequentialEncoder,
        batch_size: int,
        rngs: nnx.Rngs,
    ):
        self.sequences = encoder.transform(sequences)
        self.encoder = encoder
        self.batch_size = batch_size
        self.rngs = rngs

    def __iter__(self) -> Self:
        """Prepares for batch iteration"""

        self.__iter_init()
        return self

    def __iter_init(self) -> Self:
        """Prepares for batch iteration"""

        if getattr(self, "batch_index", 1) == 0:
            return self

        random.seed(self.seed)
        shuffled_sequences = random.sample(self.sequences, len(self.sequences))
        self.input_X, self.output_X, self.mask_X, self.nondummy_X = self.__make_dataset(
            sequences=shuffled_sequences
        )

        # Num. of batch
        self.batch_num = self.input_X.shape[1]

        # Initialize batch index
        self.batch_index = 0

        return self

    def __len__(self) -> int:
        """Returns the number of batches

        Returns:
            int: The number of batches
        """

        self.__iter_init()
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
            input_X, mask_X, nondummy_X = (
                self.input_X[:, (self.batch_index,)],
                self.mask_X[:, (self.batch_index,)],
                self.nondummy_X[:, (self.batch_index,)],
            )
            y = self.output_X[:, (self.batch_index,)]

            # Update batch index
            self.batch_index += 1

            return (input_X, mask_X, nondummy_X), y

    @property
    def seed(self) -> int:
        return jax.random.randint(self.rngs(), (1,), 0, 2**16).tolist()[0]

    def __make_dataset(
        self, sequences: list[list[int]]
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # Initialize
        input_X = [[] for _ in range(self.batch_size)]
        output_X = [[] for _ in range(self.batch_size)]
        mask_X = [[] for _ in range(self.batch_size)]
        nondummy_X = [[] for _ in range(self.batch_size)]
        num_by_row = [0 for _ in range(self.batch_size)]

        # Split
        for seq in tqdm(sequences, leave=False, desc="[MAKE DATASET]"):
            i = np.argmin(num_by_row).tolist()
            input_X[i] += seq[:-1]
            output_X[i] += seq[1:]
            mask_X[i] += [0.0] + [1.0] * (len(seq) - 2)
            nondummy_X[i] += [1] * (len(seq) - 1)
            num_by_row[i] += len(seq) - 1

        # Fill dummy values
        batch_num = max(map(len, input_X))
        input_X = [
            [self.encoder.item_num] * (batch_num - len(row)) + row for row in input_X
        ]
        output_X = [
            [self.encoder.item_num] * (batch_num - len(row)) + row for row in output_X
        ]
        mask_X = [[0.0] * (batch_num - len(row)) + row for row in mask_X]
        nondummy_X = [[0] * (batch_num - len(row)) + row for row in nondummy_X]

        # Cast to jax.Array
        input_X = jax.device_put(np.array(input_X))
        output_X = jax.device_put(np.array(output_X))
        mask_X = jax.device_put(np.array(mask_X))
        nondummy_X = jax.device_put(np.array(nondummy_X))

        return input_X, output_X, mask_X, nondummy_X
