from .curry import curry


@curry
def cond(lst, x):
    conditions = lst if not isinstance(lst, dict) else lst.items()

    for pred, fn in conditions:
        if pred(x):
            return fn(x)


@curry
def complement(pred, x):
    return not pred(x)


@curry
def both(pred1, pred2, x):
    return pred1(x) and pred2(x)


@curry
def either(pred1, pred2, x):
    return pred1(x) or pred2(x)


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
def all_(pred, it):
    return all(pred(x) for x in it)
