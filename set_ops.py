from typing import Iterable, List, TypeVar

T = TypeVar('T')

def sorted_set_op(items: Iterable[T], func) -> List[T]:
    sets = [set(item) for item in items]
    data = func(*sets)
    return sorted(data)

def sorted_intersection(*items: T) -> List[T]:
    return sorted_set_op(items, set.intersection)

def sorted_union(*items: T) -> List[T]:
    return sorted_set_op(items, set.union)

del T
