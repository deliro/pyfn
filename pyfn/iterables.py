import operator
from collections import Counter, deque
from itertools import (
    accumulate as _accumulate,
    dropwhile,
    takewhile,
    islice,
    combinations as _combinations,
    combinations_with_replacement as _combinations_with_replacement,
    permutations as _permutations,
    chain,
    zip_longest as _zip_longest,
)

from .funcs import compose, apply
from .curry import curry
from .logic import complement, either, identity, always

len_ = curry(len)


def _assert_positive(i):
    if i < 0:
        raise ValueError("Index cannot be negative (non-list sequence)")


@curry
def len_eq(x, y):
    return len(x) == y


@curry
def len_gt(x, y):
    return len(x) > y


@curry
def len_lt(x, y):
    return complement(either(len_gt(x, y), len_eq(x, y)))


@curry
def contains(e, it):
    if isinstance(it, (tuple, list, set)):
        return e in it
    return any(e == x for x in it)


@curry
def count_by(fn, x):
    return compose(Counter, map_(fn))(x)


@curry
def count(x):
    return count_by(identity, x)


@curry
def drop(n, lst):
    return lst[n:]


@curry
def drop_last(n, lst):
    return lst[:-n]


def _take_iterable(n, it):
    for i, el in enumerate(it, start=1):
        if i > n:
            return
        yield el


@curry
def take(n, it):
    try:
        return it[:n]
    except TypeError:
        return _take_iterable(n, it)


@curry
def take_last(n, it):
    try:
        return it[-n:]
    except TypeError:
        # There is no effective generator's tail traverse
        q = deque(maxlen=n)
        q.extend(it)
        return q


@curry
def uniq_by(fn, lst):
    pseudo_set = []

    for item in lst:
        applied = fn(item)
        if applied not in pseudo_set:
            pseudo_set.append(applied)
            yield item


@curry
def uniq(lst):
    return uniq_by(identity, lst)


def _adjust_iterable(fn, index, it):
    _assert_positive(index)

    for i, el in enumerate(it):
        if i == index:
            yield fn(el)
        else:
            yield el


@curry
def adjust(fn, index, it):
    if not isinstance(it, list):
        return _adjust_iterable(fn, index, it)

    new = it[:]
    new[index] = fn(new[index])
    return new


@curry
def accumulate_with(fn, it):
    return _accumulate(it, func=fn)


@curry
def accumulate(it):
    return accumulate_with(operator.add, it)


@curry
def drop_while(pred, it):
    return dropwhile(pred, it)


@curry
def take_while(pred, it):
    return takewhile(pred, it)


@curry
def slice_(start, end, it):
    if isinstance(it, (list, tuple)):
        return it[start:end]

    return islice(it, start, end)


@curry
def combinations(r, it):
    return _combinations(it, r)


@curry
def combinations_with_replacement(r, it):
    return _combinations_with_replacement(it, r)


@curry
def permutations(r, it):
    return _permutations(it, r)


@curry
def converge(fn, fn_list, x):
    return fn(*(f(x) for f in fn_list))


@curry
def aperture(n, it):
    a = list(it)

    if n > len(a):
        return []

    for i in range(len(a) - n + 1):
        yield a[i : i + n]


@curry
def sort(x):
    return sorted(x)


@curry
def sort_by(key, x):
    return sorted(x, key=key)


@curry
def reverse(x):
    try:
        return reversed(x)
    except TypeError:
        return reversed(list(x))


@curry
def zip_with(fn, it1, it2):
    return map_(apply(fn), zip(it1, it2))


@curry
def filter_(pred, it):
    return filter(pred, it)


@curry
def reject(pred, it):
    return filter_(complement(pred), it)


@curry
def find(pred, it):
    try:
        return next(filter_(pred)(it))
    except StopIteration:
        return None


@curry
def find_index(pred, it):
    for i, e in enumerate(it):
        if pred(e):
            return i
    return -1


@curry
def find_last(pred, it):
    return find(pred, reverse(it))


@curry
def find_last_index(pred, it):
    idx = -1
    for i, e in enumerate(it):
        if pred(e):
            idx = i
    return idx


@curry
def head(x):
    try:
        return next(iter(x))
    except StopIteration:
        return x.__class__() if not isinstance(x, (list, tuple)) else None


@curry
def tail(x):
    try:
        return next(reverse(x))
    except StopIteration:
        return x.__class__() if not isinstance(x, (list, tuple)) else None


@curry
def init(x):
    if isinstance(x, (list, tuple, str, bytes)):
        return x[1:]
    it = iter(x)
    next(it)
    return it


def _insert_iterable(index, v, it):
    if index != -1:
        for i, el in enumerate(it):
            if i == index:
                yield v
            yield el
    else:
        yield from it
        yield v


@curry
def insert(index, v, it):
    if index < 0 and index != -1:
        raise ValueError("Negative indices (but -1) are not supported for insert")

    if isinstance(it, list):
        new = it[:]
        if index == -1:
            new.append(v)
        else:
            new.insert(index, v)
        return new
    else:
        return _insert_iterable(index, v, it)


@curry
def prepend(v, it):
    return insert(0, v, it)


@curry
def append(v, it):
    return insert(-1, v, it)


@curry
def concat(l1, l2):
    if isinstance(l1, (tuple, list, str, bytes)):
        return l1 + l2
    return chain(l1, l2)


def nth_arg(i):
    return lambda *x: x[i]


@curry
def nth(i, it):
    try:
        return it[i]
    except TypeError:
        _assert_positive(i)

        for index, el in enumerate(it):
            if index == i:
                return el


@curry
def flatten(it):
    for el in it:
        yield from el


@curry
def zip_(it1, it2):
    return zip(it1, it2)


@curry
def zip_longest(fill_value, it1, it2):
    return _zip_longest(it1, it2, fillvalue=fill_value)


@curry
def update(i, v, lst):
    return adjust(always(v), i, lst)


@curry
def map_(fn, x):
    return map(fn, x)
