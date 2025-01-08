import jax
import jax.numpy as jnp
from flax import nnx

from flax_recsys.model.FM import FM


class FwFM(FM):
    """FwFM"""

    def __init__(
        self,
        categorical_feature_cardinalities: list[int],
        numerical_feature_num: int,
        embed_dim: int,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            categorical_feature_cardinalities=categorical_feature_cardinalities,
            numerical_feature_num=numerical_feature_num,
            embed_dim=embed_dim,
            rngs=rngs,
        )

        # Field Weight R
        m = len(categorical_feature_cardinalities) + numerical_feature_num
        self.R = nnx.Param(jax.random.normal(rngs.params(), (m, m), jnp.float32))

    def interaction_term_by_row(
        self, categorical_X_row: jax.Array, numerical_X_row: jax.Array
    ) -> jax.Array:
        V = self.generate_V_by_row(categorical_X_row, numerical_X_row)
        return jnp.sum(jnp.tril(V @ V.T, k=-1) * self.R.value)
