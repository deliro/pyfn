import math

from .curry import curry
from .logic import either, equals


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
def clamp(l, u, x):
    return max(l, min(u, x))


def inc(x):
    return x + 1


def dec(x):
    return x - 1


@curry
def modulo(x, y):
    return x % y


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
def negate(x):
    return -x
