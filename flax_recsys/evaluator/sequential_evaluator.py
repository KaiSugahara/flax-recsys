from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx
from flax_trainer.evaluator import BaseEvaluator
from tqdm.auto import tqdm

from flax_recsys.encoder.sequential_encoder import SequentialEncoder
from flax_recsys.loader.sequential_loader import SequentialLoader

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

    @staticmethod
    @nnx.jit
    def evaluate_batch(
        model: Model, Xs: tuple[jax.Array, ...], y: jax.Array
    ) -> dict[str, jax.Array]:
        # Prediction
        pred = nnx.softmax(model(*Xs))  # type: ignore
        _, top_k_indices = jax.lax.top_k(pred, k=10)

        # Cross Entropy
        logits = pred[jnp.arange(pred.shape[0]), y.reshape(-1)].reshape(-1, 1)
        ce = -jnp.log(logits + 1e-10)

        # Hit@10
        hit_10 = jax.vmap(
            (lambda indices, index: jnp.isin(index, indices)),
            in_axes=(0, 0),
            out_axes=0,
        )(top_k_indices, y)

        return {"cross_entropy": ce, "hit_10": hit_10}

    def evaluate(self, model: Model) -> tuple[float, dict[str, float]]:
        result_by_batch = [
            self.evaluate_batch(model, Xs, y) for Xs, y in tqdm(self.batches)
        ]
        result_by_metric = {
            metric_name: jnp.hstack([b[metric_name] for b in result_by_batch])
            for metric_name in set(result_by_batch[0].keys())
        }
        nondummy = jnp.hstack(
            [nondummy_X for (input_X, mask_X, nondummy_X), y in self.batches]
        )

        metrics = {
            metric_name: float((res * nondummy).sum() / nondummy.sum())
            for metric_name, res in result_by_metric.items()
        }

        return metrics["cross_entropy"], metrics
