import re

from .curry import curry


@curry
def startswith(prefix, x):
    return x.startswith(prefix)


@curry
def endswith(postfix, x):
    return x.endswith(postfix)


@curry
def split(sep, x):
    return x.split(sep)


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
def join(d, it):
    return d.join(it)


@curry
def replace(p, r, s):
    if isinstance(p, re.Pattern):
        return p.sub(r, s)
    return s.replace(p, r)


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
