"""Warmup-only choice among launch options for one small fixed-shape kernel.

The options run the same kernel source; only the launch width differs. Each
choice is made once per key on the ordinary stream and is frozen before any
CUDA graph capture, so measured samples never compile or re-select.
"""

import torch

_BEST = {}


def pick(key, default, options, launch):
    """The current launch option for ``key``; the captured step re-judges it.

    An isolated timing of a 3 us kernel proved too noisy to trust, and the
    whole run has a 15-minute limit, so nothing is timed here.
    """
    if key not in _BEST:
        if torch.cuda.is_current_stream_capturing():
            return default
        _BEST[key] = default
        # key[1] is the row count for every caller; all options are one kernel.
        register(
            key, key[1], 1, [default, *[option for option in options if option != default]],
            lambda: _BEST[key], lambda option: _BEST.__setitem__(key, option),
        )
    return _BEST[key]


class Knob:
    """One frozen-at-warmup choice that the captured decode step can re-judge."""

    def __init__(self, name, rows, weight, options, get, select, shadows=None):
        self.name, self.rows, self.weight = name, rows, weight
        self.options, self.get, self.select = options, get, select
        # Name of a knob that has no effect while this one's choice is not None.
        self.shadows = shadows


_KNOBS = {}


def register(name, rows, weight, options, get, select, shadows=None):
    _KNOBS[name] = Knob(name, rows, weight, options, get, select, shadows)


def knobs(rows):
    """Knobs of this batch size, largest expected effect first."""
    return sorted((knob for knob in _KNOBS.values() if knob.rows == rows), key=lambda knob: -knob.weight)
