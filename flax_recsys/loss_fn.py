import jax
import jax.numpy as jnp
from flax import nnx


def cross_entropy_loss(
    model: nnx.Module, Xs: tuple[jax.Array, ...], y: jax.Array
) -> jax.Array:
    # Prediction
    pred = nnx.softmax(model(*Xs))  # type: ignore

    # Cross Entropy
    logits = pred[jnp.arange(pred.shape[0]), y.reshape(-1)]
    loss = -jnp.mean(jnp.log(logits + 1e-10))

    return loss
