from functools import reduce as _reduce, partial

from .common import ArityError
from .curry import curry


@curry
def is_instance(types, obj):
    return isinstance(obj, types)


@curry
def raise_(exc):
    raise exc


@curry
def call(fn):
    return fn()


@curry
def apply(fn, x):
    return fn(*x)


@curry
def tap(fn, x):
    fn(x)
    return x


@curry
def times(fn, n):
    for i in range(n):
        yield fn(i)


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


def flip(fn):
    def inner(*args):
        try:
            a, b, *rest = args
        except ValueError:
            raise ArityError("flipped function must take at least 2 arguments")
        return curry(fn)(b, a, *rest)

    return inner


@curry
def default_to(d, x):
    if x:
        return x
    return d() if callable(d) else d
