import inspect

from .common import placeholder


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
