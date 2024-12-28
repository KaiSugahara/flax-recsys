import jax
import jax.numpy as jnp
from flax import nnx


class TwoTower(nnx.Module):
    """Two Tower"""

    def __init__(
        self,
        user_categorical_feature_indices: list[int],
        user_categorical_feature_cardinalities: list[int],
        user_numerical_feature_indices: list[int],
        user_hidden_layer_dims: list[int],
        item_categorical_feature_indices: list[int],
        item_categorical_feature_cardinalities: list[int],
        item_numerical_feature_indices: list[int],
        item_hidden_layer_dims: list[int],
        embed_dim: int,
        output_layer_dim: int,
        rngs: nnx.Rngs,
    ):
        assert len(user_categorical_feature_indices) == len(
            user_categorical_feature_cardinalities
        )
        assert len(item_categorical_feature_indices) == len(
            item_categorical_feature_cardinalities
        )

        (
            self.user_categorical_feature_indices,
            self.user_numerical_feature_indices,
            self.item_categorical_feature_indices,
            self.item_numerical_feature_indices,
        ) = (
            user_categorical_feature_indices,
            user_numerical_feature_indices,
            item_categorical_feature_indices,
            item_numerical_feature_indices,
        )

        self.user_embedders, self.item_embedders = (
            [
                nnx.Embed(num_embeddings=card, features=embed_dim, rngs=rngs)
                for card in user_categorical_feature_cardinalities
            ],
            [
                nnx.Embed(num_embeddings=card, features=embed_dim, rngs=rngs)
                for card in item_categorical_feature_cardinalities
            ],
        )

        self.user_linears, self.item_linears = (
            [
                nnx.Linear(
                    in_features=(len(user_categorical_feature_indices) * embed_dim),
                    out_features=user_hidden_layer_dims[0],
                    rngs=rngs,
                )
            ]
            + [
                nnx.Linear(
                    in_features=user_hidden_layer_dims[i - 1],
                    out_features=user_hidden_layer_dims[i],
                    rngs=rngs,
                )
                for i in range(1, len(user_hidden_layer_dims))
            ]
            + [
                nnx.Linear(
                    in_features=user_hidden_layer_dims[-1],
                    out_features=output_layer_dim,
                    rngs=rngs,
                )
            ],
            [
                nnx.Linear(
                    in_features=(len(item_categorical_feature_indices) * embed_dim),
                    out_features=item_hidden_layer_dims[0],
                    rngs=rngs,
                )
            ]
            + [
                nnx.Linear(
                    in_features=item_hidden_layer_dims[i - 1],
                    out_features=item_hidden_layer_dims[i],
                    rngs=rngs,
                )
                for i in range(1, len(item_hidden_layer_dims))
            ]
            + [
                nnx.Linear(
                    in_features=item_hidden_layer_dims[-1],
                    out_features=output_layer_dim,
                    rngs=rngs,
                )
            ],
        )

    def user_tower(self, X: jax.Array) -> jax.Array:
        X = jnp.hstack(
            [
                self.user_embedders[i](X[:, idx])
                for i, idx in enumerate(self.user_categorical_feature_indices)
            ]
            + [X[:, self.user_numerical_feature_indices]]
        )
        for linear in self.user_linears:
            X = linear(X)
        return X

    def item_tower(self, X: jax.Array) -> jax.Array:
        X = jnp.hstack(
            [
                self.item_embedders[i](X[:, idx])
                for i, idx in enumerate(self.item_categorical_feature_indices)
            ]
            + [X[:, self.item_numerical_feature_indices]]
        )
        for linear in self.item_linears:
            X = linear(X)
        return X

    def __call__(self, X: jax.Array) -> jax.Array:
        return jnp.sum(
            (self.user_tower(X) * self.item_tower(X)),
            axis=1,
            keepdims=True,
        )
