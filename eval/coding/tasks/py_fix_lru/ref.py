import functools
import time
from collections import OrderedDict


def ttl_lru(maxsize=128, ttl=None, clock=time.monotonic):
    def deco(fn):
        data = OrderedDict()
        hits = misses = 0

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            key = (args, frozenset(kwargs.items()))
            entry = data.get(key)
            if entry is not None:
                value, stored = entry
                if ttl is None or clock() - stored < ttl:
                    hits += 1
                    data.move_to_end(key)
                    return value
            misses += 1
            value = fn(*args, **kwargs)
            data[key] = (value, clock())
            data.move_to_end(key)
            if len(data) > maxsize:
                data.popitem(last=False)
            return value

        def cache_info():
            return hits, misses, len(data)

        def cache_clear():
            nonlocal hits, misses
            data.clear()
            hits = misses = 0

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper
    return deco
