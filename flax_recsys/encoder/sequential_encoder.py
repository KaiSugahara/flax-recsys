import itertools
from typing import Generic, Self, TypeVar

ITEM = TypeVar("ITEM", str, int)


class SequentialEncoder(Generic[ITEM]):
    def __init__(
        self,
    ):
        # Initialize
        self.item_map: dict[ITEM, int] = {}

    def fit(
        self,
        sequences: list[list[ITEM]],
    ) -> Self:
        # Re-Initialize
        self.item_map = {}

        # Build item_map
        unique_items = sorted(set(itertools.chain.from_iterable(sequences)))
        self.item_map = dict(zip(unique_items, range(1, len(unique_items) + 1)))

        return self

    def transform(
        self,
        sequences: list[list[ITEM]],
    ) -> list[list[int]]:
        return [
            [self.item_map[item] if item in self.item_map.keys() else 0 for item in seq]
            for seq in sequences
        ]

    def fit_transform(
        self,
        sequences: list[list[ITEM]],
    ) -> list[list[int]]:
        return self.fit(sequences=sequences).transform(sequences=sequences)

    @property
    def item_num(self) -> int:
        return len(self.item_map) + 1
