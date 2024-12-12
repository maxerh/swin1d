def cast_to_tuple(t, l=1):
    return ((t,) * l) if not isinstance(t, tuple) else t


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}