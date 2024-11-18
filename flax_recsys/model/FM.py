import jax
import jax.numpy as jnp
from flax import nnx


class FM(nnx.Module):
    """Factorization Machines"""

    def __init__(self, data_dim: int, embed_dim: int, rngs: nnx.Rngs):
        """Initialize model layer and parameters

        Args:
            data_dim (int): number of dataset dimensions.
            embed_dim (int): number of feature dimensions.
            rngs (nnx.Rngs): rng key.
        """

        self.data_dim = data_dim
        self.embedder = nnx.Embed(
            num_embeddings=data_dim, features=embed_dim, rngs=rngs
        )
        self.linear = nnx.Linear(in_features=data_dim, out_features=1, rngs=rngs)

    def __call__(self, X: jax.Array) -> jax.Array:
        # Linear Term
        linear_term_X = self.linear(X)

        # Interaction Term
        V = self.embedder.embedding.value
        interaction_term_X = (
            jnp.sum((X.dot(V)) ** 2 - (X**2).dot(V**2), axis=1) / 2
        ).reshape(-1, 1)

        return linear_term_X + interaction_term_X
