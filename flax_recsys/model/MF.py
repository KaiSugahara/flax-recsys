import jax
import jax.numpy as jnp
from flax import nnx


class MF(nnx.Module):
    """Matrix Factorization"""

    def __init__(self, user_num: int, item_num: int, embed_dim: int, rngs: nnx.Rngs):
        self.user_embedder, self.item_embedder = (
            nnx.Embed(num_embeddings=user_num, features=embed_dim, rngs=rngs),
            nnx.Embed(num_embeddings=item_num, features=embed_dim, rngs=rngs),
        )

    def __call__(self, X: jax.Array) -> jax.Array:
        return jnp.sum(
            (self.user_embedder(X[:, 0]) * self.item_embedder(X[:, 1])),
            axis=1,
            keepdims=True,
        )
