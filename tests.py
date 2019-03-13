import json
import unittest

from pyfn import *
from pyfn import __


class PyFNTestCase(unittest.TestCase):
    def test_trim(self):
        x = "  hello   "
        y = "hello"
        self.assertEqual(trim()(x), y)
        self.assertEqual(trim(x), y)

    def test_split(self):
        x = "hello world foo bar"
        y = x.split()

        self.assertEqual(split(" ", x), y)
        self.assertEqual(split(" ")(x), y)

    def test_map(self):
        x = [1, 2, 3, 4]
        y = [1, 4, 9, 16]

        self.assertEqual(list(map_(square, x)), y)
        self.assertEqual(list(map_(square)(x)), y)

    def test_lt(self):
        self.assertFalse(lt(5)(5))
        self.assertFalse(lt(6, 5))
        self.assertTrue(lt(3, 5))
        self.assertTrue(lt(3)(5))

    def test_lte(self):
        self.assertTrue(lte(5, 5))
        self.assertTrue(lte(4, 5))
        self.assertFalse(lte(6, 5))

    def test_gte(self):
        self.assertTrue(gte(6, 5))
        self.assertTrue(gte(5, 5))
        self.assertFalse(gte(4, 5))

    def test_when(self):
        fn = when(lt(__, 5), always(5))
        self.assertEqual(fn(3), 5)
        self.assertEqual(fn(4), 5)
        self.assertEqual(fn(5), 5)
        self.assertEqual(fn(6), 6)

        self.assertEqual(when(lt(__, 5), always(5), 6), 6)

    def test_unless(self):
        fn = unless(lt(__, 0), always("hello"))

        self.assertEqual(fn(5), "hello")
        self.assertEqual(fn(-1), -1)
        self.assertEqual(fn(0), "hello")

        self.assertEqual(unless(T, always("hello"), 0), 0)
        self.assertEqual(unless(F, always("hello"), 0), "hello")

    def test_lower(self):
        self.assertEqual(lower("HeLLO"), "hello")
        self.assertEqual(lower()("HeLLO"), "hello")

    def test_either(self):
        self.assertTrue(either(T, F)(None))
        self.assertTrue(either(F, T)(None))
        self.assertTrue(either(T, T)(None))
        self.assertFalse(either(F, F)(None))

    def test_both(self):
        self.assertTrue(both(T, T)(None))
        self.assertFalse(both(F)(T)(None))
        self.assertFalse(both(T)(F, None))
        self.assertFalse(both(F)(F)(None))

    def test_tail(self):
        self.assertEqual(tail([1, 2, 3]), 3)
        self.assertIsNone(tail([]))
        self.assertEqual(tail("abc"), "c")
        self.assertEqual(tail(""), "")
        self.assertEqual(tail(b"qwe"), ord("e"))
        self.assertEqual(tail(b""), b"")

        def gen():
            yield "foo"
            yield "bar"

        self.assertEqual(tail(gen()), "bar")

    def test_head(self):
        self.assertEqual(head([1, 2, 3]), 1)
        self.assertEqual(head([]), None)
        self.assertEqual(head(()), None)
        self.assertEqual(head("abc"), "a")
        self.assertEqual(head(""), "")
        self.assertEqual(head(b"abc"), ord("a"))

        def gen():
            yield "foo"
            yield "bar"

        self.assertEqual(head(gen()), "foo")

    def test_init(self):
        self.assertEqual(init(["a", "b", "c"]), ["b", "c"])
        self.assertEqual(init("abc"), "bc")

    def test_insert(self):
        x = [1, 2, 3, 4]
        self.assertEqual(insert(2, "a")(x), [1, 2, "a", 3, 4])
        self.assertEqual(x, [1, 2, 3, 4])

        f = range(4)

        self.assertEqual(list(insert(2, "a", f)), [0, 1, "a", 2, 3])
        self.assertNotIsInstance(insert(2, "a", f), list)

    def test_prepend(self):
        x = range(1, 10)
        self.assertEqual(list(prepend(0, x)), [0] + list(x))

    def test_append(self):
        x = range(10)
        y = list(x) + [10]

        self.assertEqual(list(append(10, x)), y)

        self.assertEqual(append(1, []), [1])
        self.assertEqual(append(4, [1, 2, 3]), [1, 2, 3, 4])

    def test_pipe(self):
        x = "1 2 3 4 5"
        fn = pipe(
            split(" "),
            map_(
                pipe(
                    int, add(2), subtract(__, 1), sqrt, pow_(__, 5), divide(__, 2), ceil
                )
            ),
            map_(str),
            join(","),
        )

        self.assertEqual(fn(x), "3,8,16,28,45")

    def test_replace(self):
        x = "hello"
        self.assertEqual(replace("o", "", x), "hell")

        pattern = re.compile(r"(\d+)")
        replacer = replace(pattern, r"+\1+")

        self.assertEqual(replacer("hello 99 911 world"), "hello +99+ +911+ world")

    def test_filter_and_reject(self):
        x = [1, 2, 3, 4, 5]
        self.assertEqual(list(filter_(even, x)), [2, 4])
        self.assertEqual(list(reject(even, x)), [1, 3, 5])

    def test_find(self):
        self.assertEqual(find(even, range(1, 10)), 2)
        self.assertIsNone(find(equals("hello"), range(1, 10)))
        self.assertIsNotNone(find(equals("hello"), ["hello"]))

    def test_flip(self):
        @flip
        def f(x, y):
            return x, y

        self.assertEqual(f(1, 2), (2, 1))

        @flip
        def fn(x, y, z, c):
            return (x / y) * (z / c)

        curried = fn(5, 1)
        self.assertTrue(callable(curried))

        self.assertEqual(curried(5, 1), 1)

        nested_flip = flip(curried)
        # now it's fn(y, x, c, z)
        # and (1/5) * (1/5) = 1/25
        self.assertAlmostEqual(nested_flip(5, 1), 1 / 25)

        with self.assertRaises(ArityError):

            @flip
            def f(x):
                return x

            f(5)

        # Kinda currying does not affect on flipping
        @curry
        @flip
        @curry
        @flip
        @curry
        def g(x, y, z):
            return x, y, z

        self.assertEqual(g(1, 2, 3), (1, 2, 3))

    def test_compose(self):
        # compose(f, g, h)(x) is f(g(h(x)))

        x = '["1", "2", "3"]'
        fn = compose(sum, map_(int), json.loads)

        self.assertEqual(fn(x), 6)

    def test_reduce(self):
        self.assertEqual(reduce(lambda x, y: x + y, [1, 2, 3, 4]), 10)
        self.assertEqual(reduce(lambda x, y: x + y, [1, 2, 3, 4], 1), 11)

        reducer = reduce(lambda x, y: x * y)
        self.assertTrue(callable(reducer))
        self.assertEqual(reducer([1, 2, 3, 4]), 24)
        self.assertEqual(reducer([1, 2, 3, 4], 10), 240)

    def test_cond(self):
        fn = cond(
            [
                (equals(0), always("freezes")),
                (equals(100), always("boils")),
                (T, identity),
            ]
        )

        self.assertEqual(fn(0), "freezes")
        self.assertEqual(fn(100), "boils")
        self.assertEqual(fn(50), 50)
        self.assertEqual(fn(-1), -1)

    def test_cond_dict(self):
        fn = cond(
            {equals(0): always("freezes"), equals(100): always("boils"), T: identity}
        )

        self.assertEqual(fn(0), "freezes")
        self.assertEqual(fn(100), "boils")
        self.assertEqual(fn(50), 50)
        self.assertEqual(fn(-1), -1)

    def test_if_else(self):
        fn = if_else(equals(0), always("zero"), equals(5))

        self.assertEqual(fn(0), "zero")
        self.assertTrue(fn(5))

    def test_prop(self):
        a_is_five = compose(equals(5), prop("a"))
        self.assertTrue(a_is_five({"a": 5}))
        self.assertFalse(a_is_five({"a": 6}))

        with self.assertRaises(KeyError):
            a_is_five({"b": 5})

    def test_prop_multiple(self):
        a_and_b = prop(["a", "b"])
        d = {"a": 5, "b": 6, "c": 7}
        self.assertEqual(a_and_b(d), (5, 6))

    def test_prop_or_none(self):
        a_is_five = compose(equals(5), prop_or_none("a"))
        self.assertTrue(a_is_five({"a": 5}))
        self.assertFalse(a_is_five({"b": 5}))

    def test_attr(self):
        class O:
            a = 5

        a_gt_zero = compose(gt(__, 0), attr("a"))

        self.assertTrue(a_gt_zero(O()))

        with self.assertRaises(AttributeError):
            a_gt_zero(object())

    def test_attr_or_none(self):
        self.assertIsNone(attr_or_none("a")(object()))

    def test_pick_dict(self):
        d = {"a": 5, "b": 6}
        new = pick({"a", "c"}, d)
        self.assertEqual(new["a"], 5)

        with self.assertRaises(KeyError):
            new["b"]

        with self.assertRaises(KeyError):
            new["c"]

    def test_merge_dict(self):
        d1 = {"a": 5, "b": 6}
        d2 = {"b": 7, "c": 8}

        self.assertDictEqual(merge(d1, d2), {"a": 5, "b": 7, "c": 8})

    def test_merge_all_dict(self):
        d1 = {"a": 1}
        d2 = {"b": 1}
        d3 = {"c": 1}

        self.assertDictEqual(merge_all(d1, d2, d3), {"a": 1, "b": 1, "c": 1})

    def test_sort(self):
        gen = (x for x in [3, 7, 5, 1])
        self.assertEqual(sort(gen), [1, 3, 5, 7])

    def test_sort_by(self):
        lst = [3, 7, 5, 1]
        gen = ({"value": x} for x in lst)
        fn = sort_by(prop("value"))
        self.assertEqual(fn(gen)[0]["value"], min(lst))

        # reinit generator
        gen = ({"value": x} for x in lst)
        self.assertEqual(fn(gen)[-1]["value"], max(lst))

    def test_reverse(self):
        self.assertNotIsInstance(reverse(range(10)), list)
        self.assertEqual(list(reverse(range(10))), list(range(10))[::-1])

    def test_concat(self):
        a = [1, 2, 3]
        b = [4, 5, 6]

        c = concat(a)(b)

        self.assertEqual(c, [1, 2, 3, 4, 5, 6])
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 3)

        self.assertEqual(list(concat(range(10), [1])), list(range(10)) + [1])

        self.assertEqual(concat((1, 2, 3), (4, 5, 6)), (1, 2, 3, 4, 5, 6))
        self.assertEqual(concat(b"hello ", b"world"), b"hello world")

    def test_sqrt(self):
        x = [4, 16, 81]
        y = [2, 4, 9]

        for i, j in zip(x, y):
            self.assertEqual(sqrt(i), j)

    def test_match(self):
        fn = match(r"^asdf$")

        self.assertIsNotNone(fn("asdf"))
        self.assertIsNone(fn(" asdf"))

        # Making sure flags are not ignored (occurs when recompile)
        pattern = r"[a-z]+"
        fn = match(re.compile(pattern, re.I))
        self.assertIsNotNone(fn("UPPERCASE"))
        self.assertIsNone(match(pattern, "UPPERCASE"))

    def test_all(self):
        fn = all_(even)
        self.assertTrue(fn([2, 4, 6, 8, 10]))
        self.assertFalse(fn([2, 4, 6, 7]))
        self.assertTrue(fn([]))

    def test_zip(self):
        x = ["x", "y", "z"]
        y = range(3)

        self.assertEqual(list(zip_(x, y)), [("x", 0), ("y", 1), ("z", 2)])

    def test_zip_longest(self):
        x = ["x", "y", "z"]
        y = range(2)

        self.assertEqual(
            list(zip_longest("hello", x, y)), [("x", 0), ("y", 1), ("z", "hello")]
        )

    def test_zip_dict(self):
        x = ["x", "y", "z"]
        y = range(2)

        self.assertDictEqual(zip_dict(x, y), {"x": 0, "y": 1})

    def test_zip_with(self):
        x = range(4)
        self.assertEqual(list(zip_with(add, x, x)), [0, 2, 4, 6])

    def test_zip_longest_dict(self):
        x = ["x", "y", "z"]
        y = range(2)

        self.assertDictEqual(zip_longest_dict(None, x, y), {"x": 0, "y": 1, "z": None})

    def test_aperture(self):
        x = range(5)
        self.assertEqual(list(aperture(3, x)), [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        self.assertEqual(list(aperture(6, x)), [])

    def test_clamp(self):
        fn = clamp(1, 10)
        self.assertEqual(fn(5), 5)
        self.assertEqual(fn(6), 6)
        self.assertEqual(fn(-1), 1)
        self.assertEqual(fn(11), 10)

    def test_contains(self):
        fn = contains(5)

        self.assertTrue(fn(range(10)))
        self.assertTrue(fn([1, 3, 5, 10]))

    def test_converge(self):
        mean = converge(divide, [sum, len])
        x = range(1, 8)

        self.assertEqual(mean(x), 4)

        strange_concat = converge(concat, [upper, lower])
        self.assertEqual(strange_concat("Yodel"), "YODELyodel")

    def test_count_by(self):
        x = [1, 1.1, 1.2, 1.3, 2, 3]
        fn = count_by(round)

        self.assertEqual(fn(x)[1], 4)
        self.assertEqual(fn(x)[2], 1)
        self.assertEqual(fn(x)[3], 1)

    def test_count(self):
        x = "abcdeabcdabcaba"
        r = count(x)

        self.assertEqual(r["a"], 5)
        self.assertEqual(r["b"], 4)

    def test_default_to(self):
        fn = default_to(42)
        self.assertEqual(fn(0), 42)
        self.assertEqual(fn(None), 42)
        self.assertEqual(fn(1), 1)

        fn = default_to(list)
        lst1 = fn(0)
        lst2 = fn(0)

        if lst1 is lst2:
            self.fail()

        self.assertEqual(lst1, lst2)

    def test_difference(self):
        l1 = [1, 2, 3, 4, 5, 6]
        l2 = [3, 6]

        self.assertEqual(difference(l1, l2), {1, 2, 4, 5})

    def test_startswith(self):
        fn = startswith("h")
        self.assertTrue(fn("hello"))
        self.assertTrue(fn("how are you?"))
        self.assertFalse(fn("wazzup"))

    def test_endswith(self):
        fn = endswith("?")
        self.assertTrue(fn("how are you?"))
        self.assertFalse(fn("wazzup"))

    def test_equals_by(self):
        fn = equals_by(round)
        self.assertTrue(fn(5.4, 5))
        self.assertFalse(fn(5.8, 5))

    def test_find_index(self):
        first_even_idx = find_index(even)
        self.assertEqual(first_even_idx([1, 3, 5, 8]), 3)
        self.assertEqual(first_even_idx([1, 3, 5, 7]), -1)

    def test_find_last(self):
        last_even = find_last(even)
        self.assertEqual(last_even(range(10)), 8)

    def test_find_last_index(self):
        last_even_idx = find_last_index(even)
        self.assertEqual(last_even_idx(range(10)), 8)

    def test_assoc(self):
        d = {"a": 1}
        self.assertDictEqual(assoc("b", 2, d), {"a": 1, "b": 2})
        self.assertDictEqual(d, {"a": 1})

    def test_dissoc(self):
        d = {"a": 1, "b": 2}
        self.assertDictEqual(dissoc("b", d), {"a": 1})
        self.assertDictEqual(d, {"a": 1, "b": 2})

    def test_drop(self):
        self.assertEqual(drop(1, ["foo", "bar", "baz"]), ["bar", "baz"])
        self.assertEqual(drop(2, ["foo", "bar", "baz"]), ["baz"])
        self.assertEqual(drop(3, ["foo", "bar", "baz"]), [])
        self.assertEqual(drop(4, ["foo", "bar", "baz"]), [])

    def test_drop_last(self):
        self.assertEqual(drop_last(1, ["foo", "bar", "baz"]), ["foo", "bar"])
        self.assertEqual(drop_last(2, ["foo", "bar", "baz"]), ["foo"])
        self.assertEqual(drop_last(4, ["foo", "bar", "baz"]), [])

    def test_take(self):
        self.assertEqual(take(1, ["foo", "bar", "baz"]), ["foo"])
        self.assertEqual(take(2, ["foo", "bar", "baz"]), ["foo", "bar"])
        self.assertEqual(take(5, ["foo", "bar", "baz"]), ["foo", "bar", "baz"])

        def gen():
            yield "foo"
            yield "bar"
            yield "baz"

        self.assertEqual(list(take(2, gen())), ["foo", "bar"])
        self.assertEqual(list(take(5, gen())), ["foo", "bar", "baz"])

    def test_take_last(self):
        self.assertEqual(take_last(1, ["foo", "bar", "baz"]), ["baz"])
        self.assertEqual(take_last(2, ["foo", "bar", "baz"]), ["bar", "baz"])
        self.assertEqual(take_last(3, ["foo", "bar", "baz"]), ["foo", "bar", "baz"])
        self.assertEqual(take_last(4, ["foo", "bar", "baz"]), ["foo", "bar", "baz"])

        def gen():
            yield "foo"
            yield "bar"
            yield "baz"

        self.assertEqual(list(take_last(1, gen())), ["baz"])
        self.assertEqual(list(take_last(2, gen())), ["bar", "baz"])
        self.assertEqual(list(take_last(3, gen())), ["foo", "bar", "baz"])
        self.assertEqual(list(take_last(4, gen())), ["foo", "bar", "baz"])

    def test_times(self):
        self.assertEqual(list(times(add(1), 5)), [1, 2, 3, 4, 5])

    def test_to_pairs(self):
        self.assertEqual(list(to_pairs({"a": 1, "b": 2})), [("a", 1), ("b", 2)])

    def test_union(self):
        self.assertEqual(union([1, 2, 3], [3, 4]), {1, 2, 3, 4})

    def test_intersection(self):
        self.assertEqual(intersection([1, 2, 3], [2, 3, 4]), {2, 3})

    def test_uniq_by(self):
        self.assertEqual(list(uniq_by(round, [1.1, 1.2, 1.3, 2])), [1.1, 2])

    def test_uniq(self):
        self.assertEqual(list(uniq([1, 1, 2, 1, 3, 1, 3])), [1, 2, 3])

    def test_adjust(self):
        self.assertEqual(adjust(add(10), 1, [1, 2, 3]), [1, 12, 3])
        self.assertEqual(list(adjust(add(10), 0, range(3))), [10, 1, 2])

    def test_update(self):
        self.assertEqual(update(1, 10, [1, 2, 3]), [1, 10, 3])

    def test_modulo(self):
        self.assertEqual(modulo(17)(3), 2)

    def test_nth(self):
        self.assertEqual(nth(2, map_(add(1), range(10))), 3)

    def test_accumulate(self):
        self.assertEqual(list(accumulate([1, 2, 3])), [1, 3, 6])

    def test_accumulate_with(self):
        fn = accumulate_with(lambda *a: join("/", a))
        self.assertEqual(
            list(fn(["usr", "local", "bin"])), ["usr", "usr/local", "usr/local/bin"]
        )

    def test_prop_satisfies(self):
        fn = prop_satisfies(lt(__, 10), "a")
        self.assertTrue(fn({"a": 5}))
        self.assertFalse(fn({"a": 11}))

        with self.assertRaises(KeyError):
            fn({"b": 5})

    def test_is_instance(self):
        is_str = is_instance(str)
        is_str_or_list = is_instance((str, list))

        self.assertTrue(is_str("hello"))
        self.assertFalse(is_str(1))

        self.assertTrue(is_str_or_list([1, 2, 3]))
        self.assertTrue(is_str_or_list("hello"))
        self.assertFalse(is_str_or_list(123))

    def test_drop_while(self):
        first_odd = drop_while(even)

        self.assertEqual(list(first_odd([2, 4, 6, 8, 10, 11, 12, 13])), [11, 12, 13])

    def test_take_while(self):
        take_b4_first_odd = take_while(even)
        self.assertEqual(list(take_b4_first_odd([2, 4, 6, 7, 8, 9])), [2, 4, 6])

    def test_slice(self):
        self.assertEqual(slice_(1, 3)([0, 1, 2, 3, 4]), [1, 2])

        self.assertEqual(list(slice_(1, 3)(range(5))), [1, 2])

    def test_raise(self):
        with self.assertRaises(ValueError):
            raise_(ValueError("hello"))

    def test_combinations(self):
        self.assertEqual(
            list(combinations(3, range(4))),
            [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        )

    def test_combinations_with_replacement(self):
        self.assertEqual(
            list(combinations_with_replacement(2, "ABC")),
            [tuple(i) for i in ["AA", "AB", "AC", "BB", "BC", "CC"]],
        )

    def test_permutations(self):
        self.assertEqual(
            list(permutations(2, range(4))),
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
            ],
        )

    def test_find_re(self):
        self.assertEqual(
            list(find_re(r"(\d+)", "2_3_4_5")), compose(list, map_(tuple))("2345")
        )

    def test_flatten(self):
        self.assertEqual(list(flatten([(1,), (2,), (3,)])), [1, 2, 3])

    def test_mean(self):
        def f():
            yield from range(11)

        self.assertEqual(mean(f()), 5)
        self.assertEqual(mean([2, 3, 4]), 3)


def benchmarks():
    from time import time
    import math

    inp = "1 2 3 4 5"
    result = "3,8,16,28,45"
    n = 3000

    functional = pipe(
        split(" "),
        map_(
            pipe(
                int,
                add(2),
                subtract(__, 1),
                sqrt,
                pow_(__, 5),
                divide(__, 2),
                ceil,
                str,
            )
        ),
        join(","),
    )

    def imperative(x):
        lst = x.split()
        tmp = []

        for el in lst:
            number = int(el)
            number += 2
            number -= 1
            number = number ** 0.5
            number = number ** 5
            number /= 2
            number = math.ceil(number)
            tmp.append(str(number))
        return ",".join(tmp)

    assert functional(inp) == imperative(inp) == result

    start = time()
    for _ in range(n):
        functional(inp)
    print("Functional:", time() - start)

    start = time()
    for _ in range(n):
        imperative(inp)
    print("Imperative:", time() - start)


if __name__ == "__main__":
    benchmarks()
    unittest.main()
