import jax
import jax.numpy as jnp
from flax import nnx


class FM(nnx.Module):
    """FM"""

    def __init__(
        self,
        categorical_feature_cardinalities: list[int],
        numerical_feature_num: int,
        embed_dim: int,
        rngs: nnx.Rngs,
    ):
        (
            self.categorical_feature_cardinalities,
            self.numerical_feature_num,
        ) = (
            categorical_feature_cardinalities,
            numerical_feature_num,
        )

        # Coefficient b for Linear Term
        self.b = {
            "categorical": {
                field_i: nnx.Embed(num_embeddings=car, features=1, rngs=rngs)
                for field_i, car in enumerate(categorical_feature_cardinalities)
            },
            "numerical": {
                field_i: nnx.Embed(num_embeddings=1, features=1, rngs=rngs)
                for field_i in range(numerical_feature_num)
            },
        }

        # Embedding Matrix W for Interaction Term
        self.W = {
            "categorical": {
                field_i: nnx.Embed(num_embeddings=car, features=embed_dim, rngs=rngs)
                for field_i, car in enumerate(categorical_feature_cardinalities)
            },
            "numerical": {
                field_i: nnx.Embed(num_embeddings=1, features=embed_dim, rngs=rngs)
                for field_i in range(numerical_feature_num)
            },
        }

        # Bias b0
        self.b0 = nnx.Param(jax.random.normal(rngs.params(), (1,), jnp.float32))

    def __call__(self, *Xs: jax.Array) -> jax.Array:
        categorical_X, numerical_X = Xs
        return jax.vmap(
            lambda categorical_X_row, numerical_X_row: (
                self.bias_term_by_row(categorical_X_row, numerical_X_row)
                + self.interaction_term_by_row(categorical_X_row, numerical_X_row)
            ),
            in_axes=(0),
            out_axes=(0),
        )(categorical_X, numerical_X).reshape(-1, 1)

    def interaction_term_by_row(
        self, categorical_X_row: jax.Array, numerical_X_row: jax.Array
    ) -> jax.Array:
        V = self.generate_V_by_row(categorical_X_row, numerical_X_row)
        return (
            (jnp.linalg.norm(V.sum(axis=0)) ** 2)
            - (jnp.linalg.norm(V, axis=1) ** 2).sum()
        ) / 2

    def bias_term_by_row(
        self, categorical_X_row: jax.Array, numerical_X_row: jax.Array
    ) -> jax.Array:
        return jnp.vstack(
            [
                self.b["categorical"][i](value)
                for i, value in enumerate(categorical_X_row)
            ]
            + [
                self.b["numerical"][i].embedding[0] * value
                for i, value in enumerate(numerical_X_row)
            ]
            + [self.b0.value]
        ).sum()

    def generate_V_by_row(
        self, categorical_X_row: jax.Array, numerical_X_row: jax.Array
    ) -> jax.Array:
        return jnp.vstack(
            [
                self.W["categorical"][i](value)
                for i, value in enumerate(categorical_X_row)
            ]
            + [
                self.W["numerical"][i].embedding[0] * value
                for i, value in enumerate(numerical_X_row)
            ]
        )
