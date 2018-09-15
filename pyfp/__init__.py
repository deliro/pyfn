import inspect
import math
import operator
import re
from collections import Counter
from functools import partial, reduce as _reduce
from itertools import (
    chain,
    zip_longest as _zip_longest,
    accumulate as _accumulate,
    dropwhile,
    takewhile,
    combinations as _combinations,
    combinations_with_replacement as _combinations_with_replacement,
    permutations as _permutations,
    islice,
)


class ArityError(Exception):
    pass


class PickProxy:
    def __init__(self, obj, attrs):
        self.__obj = obj
        self.__attrs = attrs

    def __getattr__(self, item):
        if item in self.__attrs:
            return getattr(self.__obj, item)
        raise AttributeError()


class MergeProxy:
    def __init__(self, obj, other):
        self.__objs = [other, obj]

    def __getattr__(self, item):
        for obj in self.__objs:
            try:
                return getattr(obj, item)
            except AttributeError:
                pass
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(self.__class__.__name__, item)
        )


empty = object()
placeholder = __ = object()


def curry(fn, received=None):
    """
    Makes curried version of a function

    These are equivalents:
        @curry
        def f(x, y, z):
            return x, y, z

        - f(1, 2, 3)
        - f(1)(2, 3)
        - f(1)(2)(3)
        - f(__, 2, 3)(1)
        - f(__, 2, __)(1, 3)

    Thanks ramda.js and MIT license for this (almost) copy-paste
    :type received: Already taken arguments
    :type fn: A function being curried
    """

    length = len(inspect.signature(fn).parameters)
    if received is None:
        received = []

    def inner(*args):
        combined = []
        combined_idx = 0
        args_idx = 0
        left = length

        while combined_idx < len(received) or args_idx < len(args):
            if (
                combined_idx < len(received)
                and received[combined_idx] is not placeholder
                or args_idx >= len(args)
            ):
                result = received[combined_idx]
            else:
                result = args[args_idx]
                args_idx += 1

            combined.append(result)

            if result is not placeholder:
                left -= 1
            combined_idx += 1

        return fn(*combined) if left <= 0 else curry(fn, combined)

    return inner


len_ = curry(len)


def _assert_positive(i):
    if i < 0:
        raise ValueError("Index cannot be negative (non-list sequence)")


def pipe(*fns):
    if not fns:
        raise ValueError("Pipe must contains at least one function")

    def inner(*args, **kwargs):
        for i, fn in enumerate(fns):
            if i == 0:
                result = fn(*args, **kwargs)
            else:
                result = fn(result)
        return result

    return inner


def compose(*fns):
    return pipe(*reversed(fns))


def reduce(*args):
    if len(args) > 1:
        return _reduce(*args)

    return partial(_reduce, args[0])


@curry
def cond(lst, x):
    conditions = lst if not isinstance(lst, dict) else lst.items()

    for pred, fn in conditions:
        if pred(x):
            return fn(x)


@curry
def split(sep, x):
    return x.split(sep)


@curry
def map_(fn, x):
    return map(fn, x)


@curry
def trim(x):
    return x.strip()


@curry
def lower(x):
    return x.lower()


@curry
def upper(x):
    return x.upper()


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
def join(d, it):
    return d.join(it)


@curry
def replace(p, r, s):
    if isinstance(p, re.Pattern):
        return p.sub(r, s)
    return s.replace(p, r)


@curry
def filter_(pred, it):
    return filter(pred, it)


@curry
def complement(pred, x):
    return not pred(x)


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
def both(pred1, pred2, x):
    return pred1(x) and pred2(x)


@curry
def either(pred1, pred2, x):
    return pred1(x) or pred2(x)


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
def lt(x, y):
    return x < y


@curry
def gt(x, y):
    return x > y


@curry
def lte(x, y):
    return either(lt(x), equals(x))(y)


@curry
def gte(x, y):
    return either(gt(x), equals(x))(y)


@curry
def attr(name, obj):
    return getattr(obj, name)


@curry
def prop(name, dct):
    if isinstance(name, list):
        return tuple(dct[x] for x in name)
    return dct[name]


@curry
def attr_or_none(name, obj):
    return getattr(obj, name, None)


@curry
def prop_or_none(name, dct):
    return dct.get(name)


@curry
def attr_eq(name, v, obj):
    return compose(equals(v), attr(name))(obj)


@curry
def prop_eq(name, v, dct):
    return compose(equals(v), prop(name))(dct)


@curry
def head(x):
    try:
        return x[0]
    except IndexError:
        return type(x)() if not isinstance(x, (list, tuple)) else None


@curry
def tail(x):
    try:
        return x[-1]
    except IndexError:
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


def flip(fn):
    def inner(*args):
        try:
            a, b, *rest = args
        except ValueError:
            raise ArityError("flipped function must take at least 2 arguments")
        return curry(fn)(b, a, *rest)

    return inner


@curry
def add(x, y):
    return x + y


@curry
def subtract(x, y):
    return x - y


@curry
def multiply(x, y):
    return x * y


@curry
def divide(x, y):
    return x / y


@curry
def divide_int(x, y):
    return x // y


@curry
def ceil(x):
    return math.ceil(x)


@curry
def negate(x):
    return -x


def always(v, *_):
    return lambda *_: v


def T(*_):
    return True


def F(*_):
    return False


def identity(v, *_):
    return v


@curry
def equals_by(pred, x, y):
    return pred(x) == pred(y)


@curry
def equals(v, x):
    return v == x


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
def if_else(pred, pos_f, neg_f, x):
    return pos_f(x) if pred(x) else neg_f(x)


@curry
def when(pred, fn, v):
    return if_else(pred, fn, identity, v)


@curry
def unless(pred, fn, v):
    return when(complement(pred), fn, v)


@curry
def pow_(n, b):
    return math.pow(n, b)


@curry
def square(b):
    return pow_(b, 2)


@curry
def cube(b):
    return pow_(b, 2)


@curry
def sqrt(b):
    return pow_(b, 0.5)


@curry
def even(x):
    return x % 2 == 0


@curry
def pick(attrs, obj):
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if k in attrs}
    return PickProxy(obj, attrs)


@curry
def merge(obj, other):
    if isinstance(obj, dict):
        new = {}
        new.update(obj)
        new.update(other)
        return new
    return MergeProxy(obj, other)


def merge_all(*args):
    if len(args) == 1:
        return MergeProxy(args[0], args[0])
    return reduce(merge, args)


@curry
def match(pattern, s):
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    return pattern.match(s)


@curry
def find_re(pattern, s):
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    for m in pattern.finditer(s):
        yield m.groups()


@curry
def flatten(it):
    for el in it:
        yield from el


@curry
def apply(fn, x):
    return fn(*x)


@curry
def all_(pred, it):
    return all(pred(x) for x in it)


@curry
def zip_(it1, it2):
    return zip(it1, it2)


@curry
def zip_longest(fill_value, it1, it2):
    return _zip_longest(it1, it2, fillvalue=fill_value)


@curry
def zip_dict(keys, values):
    return dict(zip(keys, values))


@curry
def zip_longest_dict(fill_value, keys, values):
    return dict(zip_longest(fill_value, keys, values))


@curry
def zip_with(fn, it1, it2):
    return map_(apply(fn), zip(it1, it2))


@curry
def aperture(n, it):
    a = list(it)

    if n > len(a):
        return []

    for i in range(len(a) - n + 1):
        yield a[i : i + n]


@curry
def clamp(l, u, x):
    return max(l, min(u, x))


@curry
def contains(e, it):
    if isinstance(it, (tuple, list, set)):
        return e in it
    return any(e == x for x in it)


@curry
def converge(fn, fn_list, x):
    return fn(*(f(x) for f in fn_list))


@curry
def count_by(fn, x):
    return compose(Counter, map_(fn))(x)


@curry
def count(x):
    return count_by(identity, x)


def inc(x):
    return x + 1


def dec(x):
    return x - 1


@curry
def default_to(d, x):
    if x:
        return x
    return d() if callable(d) else d


@curry
def difference(s1, s2):
    return set(s1) - set(s2)


@curry
def intersection(s1, s2):
    return set(s1) & set(s2)


@curry
def union(s1, s2):
    return set(s1) | set(s2)


@curry
def startswith(prefix, x):
    return x.startswith(prefix)


@curry
def endswith(postfix, x):
    return x.endswith(postfix)


@curry
def assoc(key, value, dct):
    new = dct.copy()
    new[key] = value
    return new


@curry
def dissoc(key, dct):
    new = dct.copy()
    del new[key]
    return new


@curry
def drop(n, lst):
    return lst[n:]


@curry
def drop_last(n, lst):
    return lst[:-n]


@curry
def take(n, lst):
    return lst[:n]


@curry
def take_last(n, lst):
    return lst[-n:]


@curry
def tap(fn, x):
    fn(x)
    return x


@curry
def times(fn, n):
    for i in range(n):
        yield fn(i)


@curry
def to_pairs(dct):
    return dct.items()


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
def update(i, v, lst):
    return adjust(always(v), i, lst)


@curry
def modulo(x, y):
    return x % y


@curry
def accumulate_with(fn, it):
    return _accumulate(it, func=fn)


@curry
def accumulate(it):
    return accumulate_with(operator.add, it)


@curry
def prop_satisfies(pred, name, dct):
    return pred(dct[name])


@curry
def is_instance(types, obj):
    return isinstance(obj, types)


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
def raise_(exc):
    raise exc


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
def call(fn):
    return fn()


@curry
def mean(it):
    try:
        l = len(it)
    except TypeError:
        iterator = iter(it)
        s = next(iterator)
        l = 1
        for l, el in enumerate(iterator, start=2):
            s += el
        return s / l
    else:
        return sum(it) / l
