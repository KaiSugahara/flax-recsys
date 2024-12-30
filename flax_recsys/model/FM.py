import jax
import jax.numpy as jnp
from flax import nnx


class FM(nnx.Module):
    """FM"""

    def __init__(
        self,
        user_categorical_feature_indices: list[int],
        user_categorical_feature_cardinalities: list[int],
        user_numerical_feature_indices: list[int],
        item_categorical_feature_indices: list[int],
        item_categorical_feature_cardinalities: list[int],
        item_numerical_feature_indices: list[int],
        embed_dim: int,
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
            self.user_categorical_feature_cardinalities,
            self.user_numerical_feature_indices,
            self.item_categorical_feature_indices,
            self.item_categorical_feature_cardinalities,
            self.item_numerical_feature_indices,
        ) = (
            user_categorical_feature_indices,
            user_categorical_feature_cardinalities,
            user_numerical_feature_indices,
            item_categorical_feature_indices,
            item_categorical_feature_cardinalities,
            item_numerical_feature_indices,
        )

        # Embedders for Linear Term
        self.user_lin_cat_embedders = [
            nnx.Embed(
                num_embeddings=car,
                features=1,
                rngs=rngs,
            )
            for car in user_categorical_feature_cardinalities
        ]
        self.user_lin_num_embedder = (
            nnx.Embed(
                num_embeddings=len(user_numerical_feature_indices),
                features=1,
                rngs=rngs,
            )
            if len(user_numerical_feature_indices) > 0
            else None
        )
        self.item_lin_cat_embedders = [
            nnx.Embed(
                num_embeddings=car,
                features=1,
                rngs=rngs,
            )
            for car in item_categorical_feature_cardinalities
        ]
        self.item_lin_num_embedder = (
            nnx.Embed(
                num_embeddings=len(item_numerical_feature_indices),
                features=1,
                rngs=rngs,
            )
            if len(item_numerical_feature_indices) > 0
            else None
        )

        # Embedders for Interaction Term
        self.user_int_cat_embedders = [
            nnx.Embed(
                num_embeddings=car,
                features=embed_dim,
                rngs=rngs,
            )
            for car in user_categorical_feature_cardinalities
        ]
        self.user_int_num_embedder = (
            nnx.Embed(
                num_embeddings=len(user_numerical_feature_indices),
                features=embed_dim,
                rngs=rngs,
            )
            if len(user_numerical_feature_indices) > 0
            else None
        )
        self.item_int_cat_embedders = [
            nnx.Embed(
                num_embeddings=car,
                features=embed_dim,
                rngs=rngs,
            )
            for car in item_categorical_feature_cardinalities
        ]
        self.item_int_num_embedder = (
            nnx.Embed(
                num_embeddings=len(item_numerical_feature_indices),
                features=embed_dim,
                rngs=rngs,
            )
            if len(item_numerical_feature_indices) > 0
            else None
        )

        # Bias
        self.bias = nnx.Embed(
            num_embeddings=1,
            features=1,
            rngs=rngs,
        )

    def __call__(self, X: jax.Array) -> jax.Array:
        interaction_term = (
            jnp.sum(
                sum(
                    [
                        self.user_int_cat_embedders[i](X[:, index])
                        for i, index in enumerate(self.user_categorical_feature_indices)
                    ]
                    + [
                        self.user_int_num_embedder.embedding[i] * X[:, (index,)]
                        for i, index in enumerate(self.user_numerical_feature_indices)
                    ]
                    + [
                        self.item_int_cat_embedders[i](X[:, index])
                        for i, index in enumerate(self.item_categorical_feature_indices)
                    ]
                    + [
                        self.item_int_num_embedder.embedding[i] * X[:, (index,)]
                        for i, index in enumerate(self.item_numerical_feature_indices)
                    ]
                )
                ** 2,
                axis=1,
                keepdims=True,
            )
            - sum(
                [
                    jnp.sum(
                        self.user_int_cat_embedders[i](X[:, index]) ** 2,
                        axis=1,
                        keepdims=True,
                    )
                    for i, index in enumerate(self.user_categorical_feature_indices)
                ]
                + [
                    jnp.sum(
                        (self.user_int_num_embedder.embedding[i] * X[:, (index,)]) ** 2,
                        axis=1,
                        keepdims=True,
                    )
                    for i, index in enumerate(self.user_numerical_feature_indices)
                ]
                + [
                    jnp.sum(
                        self.item_int_cat_embedders[i](X[:, index]) ** 2,
                        axis=1,
                        keepdims=True,
                    )
                    for i, index in enumerate(self.item_categorical_feature_indices)
                ]
                + [
                    jnp.sum(
                        (self.item_int_num_embedder.embedding[i] * X[:, (index,)]) ** 2,
                        axis=1,
                        keepdims=True,
                    )
                    for i, index in enumerate(self.item_numerical_feature_indices)
                ]
            )
        ) / 2
        bias_term = (
            sum(
                [
                    self.user_lin_cat_embedders[i](X[:, index])
                    for i, index in enumerate(self.user_categorical_feature_indices)
                ]
                + [
                    self.user_lin_num_embedder.embedding[i]
                    for i, index in enumerate(self.user_numerical_feature_indices)
                ]
                + [
                    self.item_lin_cat_embedders[i](X[:, index])
                    for i, index in enumerate(self.item_categorical_feature_indices)
                ]
                + [
                    self.item_lin_num_embedder.embedding[i]
                    for i, index in enumerate(self.item_numerical_feature_indices)
                ]
            )
            + self.bias.embedding[0]
        )

        return interaction_term + bias_term
