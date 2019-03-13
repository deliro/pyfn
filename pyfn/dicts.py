from .funcs import compose, reduce, is_instance
from .curry import curry
from .iterables import zip_longest
from .logic import equals


_is_dict = is_instance(dict)


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
def pick(attrs, obj):
    if not _is_dict(obj):
        raise TypeError("pick works only with dicts")
    return {k: v for k, v in obj.items() if k in attrs}


@curry
def merge(obj, other):
    if not _is_dict(obj) or not _is_dict(other):
        raise TypeError("merge works only with dicts")

    new = {}
    new.update(obj)
    new.update(other)
    return new


def merge_all(*args):
    if len(args) == 1:
        return args[0].copy()
    return reduce(merge, args)


@curry
def zip_dict(keys, values):
    return dict(zip(keys, values))


@curry
def zip_longest_dict(fill_value, keys, values):
    return dict(zip_longest(fill_value, keys, values))


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
def to_pairs(dct):
    return dct.items()


@curry
def prop_satisfies(pred, name, dct):
    return pred(dct[name])
