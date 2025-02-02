from typing import TypeVar

T = TypeVar("T")


def split_array_into_windows(arr: list[T], window_size: int) -> list[list[T]]:
    return [arr[i : (i + window_size)] for i in range(len(arr) - window_size + 1)]
