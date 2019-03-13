from .curry import curry
from .funcs import is_instance


_is_set = is_instance(set)


def _ensure_sets(s1, s2):
    if not _is_set(s1):
        s1 = set(s1)
    if not _is_set(s2):
        s2 = set(s2)
    return s1, s2


@curry
def difference(s1, s2):
    s1, s2 = _ensure_sets(s1, s2)
    return s1 - s2


@curry
def intersection(s1, s2):
    s1, s2 = _ensure_sets(s1, s2)
    return s1 & s2


@curry
def union(s1, s2):
    s1, s2 = _ensure_sets(s1, s2)
    return s1 | s2
