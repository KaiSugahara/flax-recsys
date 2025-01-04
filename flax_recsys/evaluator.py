import jax
import jax.numpy as jnp
from flax import linen as nn
from flax_trainer.evaluator import BaseEvaluator


class GeneralRatingEvaluator(BaseEvaluator):
    """

    Attributes:
        categorical_X (jax.Array): Categorical features in the testing data.
        numerical_X (jax.Array): Numerical features in the testing data.
        y (jax.Array): Rating in the testing data.
    """

    def __init__(
        self,
        categorical_X: jax.Array,
        numerical_X: jax.Array,
        y: jax.Array,
    ):
        self.categorical_X = categorical_X
        self.numerical_X = numerical_X
        self.y = y

    def evaluate(self, model: nn.Module) -> tuple[float, dict[str, float]]:
        """Calculates test loss and metrics.

        Args:
            model (nn.Module): The Flax model you are training.

        Returns:
            float: The test loss.
            dict[str, float]: The test metrics.
        """

        def calc_mse(Xs: tuple[jax.Array, ...], y: jax.Array) -> jax.Array:
            # Prediction
            pred = model(*Xs)

            # MSE
            loss = jnp.mean((pred - y) ** 2)

            return loss

        # MSE
        mse = float(calc_mse((self.categorical_X, self.numerical_X), self.y))

        return mse, {"mse": mse}
