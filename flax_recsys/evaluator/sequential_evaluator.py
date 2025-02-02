from typing import Generic, TypeVar

import jax.numpy as jnp
from flax import nnx
from flax_trainer.evaluator import BaseEvaluator
from tqdm.auto import tqdm

from flax_recsys.encoder.sequential_encoder import SequentialEncoder
from flax_recsys.loader.sequential_loader import SequentialLoader
from flax_recsys.loss_fn import cross_entropy_loss

T = TypeVar("T", str, int)
Model = TypeVar("Model", bound=nnx.Module)


class SequentialEvaluator(BaseEvaluator, Generic[T, Model]):
    def __init__(
        self,
        sequences: list[list[T]],
        encoder: SequentialEncoder,
        batch_size: int,
    ):
        self.sequences = encoder.transform(sequences)
        self.encoder = encoder

        self.batches = list(
            SequentialLoader(
                sequences=sequences,
                encoder=encoder,
                batch_size=batch_size,
                rngs=nnx.Rngs(0),
            )
        )

    def evaluate(self, model: Model) -> tuple[float, dict[str, float]]:
        calc_cross_entropy_loss = nnx.jit(cross_entropy_loss)

        # Cross entropy
        ce = float(
            jnp.stack(
                [calc_cross_entropy_loss(model, Xs, y) for Xs, y in tqdm(self.batches)]
            ).mean()
        )

        return ce, {"cross_entropy": ce}
