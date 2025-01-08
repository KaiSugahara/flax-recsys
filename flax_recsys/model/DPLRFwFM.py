import jax
import jax.numpy as jnp
from flax import nnx

from flax_recsys.model.FM import FM


class DPLRFwFM(FM):
    """DPLRFwFM"""

    def __init__(
        self,
        categorical_feature_cardinalities: list[int],
        numerical_feature_num: int,
        embed_dim: int,
        rho: int,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            categorical_feature_cardinalities=categorical_feature_cardinalities,
            numerical_feature_num=numerical_feature_num,
            embed_dim=embed_dim,
            rngs=rngs,
        )

        # Parameter U
        m = len(categorical_feature_cardinalities) + numerical_feature_num
        self.U = nnx.Param(jax.random.normal(rngs.params(), (rho, m), jnp.float32))

        # Parameter e
        self.e = nnx.Param(jax.random.normal(rngs.params(), (rho,), jnp.float32))

    def interaction_term_by_row(
        self, categorical_X_row: jax.Array, numerical_X_row: jax.Array
    ) -> jax.Array:
        V = self.generate_V_by_row(categorical_X_row, numerical_X_row)
        P = self.U.value @ V
        return (self.d @ jnp.linalg.norm(V, axis=1)) + (
            self.e.value @ jnp.linalg.norm(P, axis=1)
        )

    @property
    def d(self) -> jax.Array:
        return -jnp.diagonal(self.U.value.T @ jnp.diag(self.e.value) @ self.U.value)

    @property
    def R(self) -> jax.Array:
        return self.U.value.T @ jnp.diag(self.e.value) @ self.U.value + jnp.diag(self.d)
