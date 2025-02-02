import jax
from flax import nnx

from flax_recsys.toolbox import split_array_into_windows


class GRU4Rec(nnx.Module):
    """GRU4Rec"""

    def __init__(
        self,
        item_num: int,
        embed_dim: int,
        gru_layer_dims: list[int],
        ff_layer_dims: list[int],
        output_layer_dim: int,
        max_batch_size: int,
        rngs: nnx.Rngs,
    ):
        self.max_batch_size = max_batch_size
        self.gru_layer_dims = gru_layer_dims

        self.item_embedder = nnx.Embed(
            num_embeddings=item_num, features=embed_dim, rngs=rngs
        )

        self.GRUCells = []
        self.carries = []
        for in_features, hidden_features in split_array_into_windows(
            [embed_dim] + gru_layer_dims, window_size=2
        ):
            self.GRUCells.append(
                nnx.GRUCell(
                    in_features=in_features, hidden_features=hidden_features, rngs=rngs
                )
            )
            self.carries.append(
                nnx.Variable(
                    self.GRUCells[-1].initialize_carry(
                        (max_batch_size, hidden_features), rngs=rngs
                    )
                )
            )

        self.ff_linears = [
            nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
            for in_features, out_features in split_array_into_windows(
                ([embed_dim] + gru_layer_dims)[-1:] + ff_layer_dims, window_size=2
            )
        ]

        self.output_linear = nnx.Linear(
            in_features=([embed_dim] + gru_layer_dims + ff_layer_dims)[-1],
            out_features=output_layer_dim,
            rngs=rngs,
        )

    def __call__(self, *Xs: jax.Array) -> jax.Array:
        input_X, mask_X, nondummy_X = Xs

        # Embed Layer
        y = self.item_embedder(input_X[:, 0])

        # GRU Layer(s)
        for i, _ in enumerate(self.GRUCells):
            carry = self.carries[i].value[: input_X.shape[0]] * mask_X
            self.carries[i].value, y = self.GRUCells[i](carry, y)

        # Feedforward Layer(s)
        for linear in self.ff_linears:
            y = linear(y)
            y = nnx.relu(y)

        # Output Layer
        y = self.output_linear(y)

        return y
