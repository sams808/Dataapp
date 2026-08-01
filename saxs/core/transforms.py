"""Axis transforms for SAXS representations (Guinier, Kratky, Porod, ...)."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

AXIS_TRANSFORMS: Dict[str, Tuple[str, Callable]] = {
    "Q": (r"q [$\AA^{-1}$]", lambda v: v),
    "Q^2": (r"q$^2$ [$\AA^{-2}$]", lambda v: v ** 2),
    "Q^3": (r"q$^3$ [$\AA^{-3}$]", lambda v: v ** 3),
    "Q^4": (r"q$^4$ [$\AA^{-4}$]", lambda v: v ** 4),
    "sqrt(Q)": (r"$\sqrt{q}$ [$\AA^{-1/2}$]", lambda v: np.sqrt(np.clip(v, 0.0, None))),
    "1/Q": (r"1/q [$\AA$]", lambda v: 1.0 / np.clip(v, 1e-300, None)),
    "ln(Q)": (r"ln(q)", lambda v: np.log(np.clip(v, 1e-300, None))),
    "log10(Q)": (r"log$_{10}$(q)", lambda v: np.log10(np.clip(v, 1e-300, None))),
    "ln(Q^2)": (r"ln(q$^2$)", lambda v: np.log(np.clip(v ** 2, 1e-300, None))),
    "log10(Q^2)": (r"log$_{10}$(q$^2$)", lambda v: np.log10(np.clip(v ** 2, 1e-300, None))),
}

Y_AXIS_TRANSFORMS: Dict[str, Tuple[str, Callable]] = {
    "I": ("I(q)", lambda q, I: I),
    "I*Q": ("I(q) q", lambda q, I: I * q),
    "I*Q^2": (r"I(q) q$^2$", lambda q, I: I * q ** 2),
    "I*Q^3": (r"I(q) q$^3$", lambda q, I: I * q ** 3),
    "I*Q^4": (r"I(q) q$^4$", lambda q, I: I * q ** 4),
    "I/Q": ("I(q) / q", lambda q, I: I / np.clip(q, 1e-300, None)),
    "sqrt(I)": (r"$\sqrt{I(q)}$", lambda q, I: np.sqrt(np.clip(I, 0.0, None))),
    "ln(I)": ("ln(I)", lambda q, I: np.log(np.clip(I, 1e-300, None))),
    "log10(I)": (r"log$_{10}$(I)", lambda q, I: np.log10(np.clip(I, 1e-300, None))),
    "ln(I*Q^2)": (r"ln(I q$^2$)", lambda q, I: np.log(np.clip(I * q ** 2, 1e-300, None))),
    "log10(I*Q^2)": (r"log$_{10}$(I q$^2$)", lambda q, I: np.log10(np.clip(I * q ** 2, 1e-300, None))),
    "ln(I*Q^4)": (r"ln(I q$^4$)", lambda q, I: np.log(np.clip(I * q ** 4, 1e-300, None))),
    "log10(I*Q^4)": (r"log$_{10}$(I q$^4$)", lambda q, I: np.log10(np.clip(I * q ** 4, 1e-300, None))),
}

# Named plot presets: (x transform, y transform, xlog, ylog)
PLOT_PRESETS: Dict[str, Tuple[str, str, bool, bool]] = {
    "log-log": ("Q", "I", True, True),
    "lin-lin": ("Q", "I", False, False),
    "Guinier (ln I vs q^2)": ("Q^2", "ln(I)", False, False),
    "Kratky (I q^2 vs q)": ("Q", "I*Q^2", False, False),
    "Porod (I q^4 vs q)": ("Q", "I*Q^4", False, False),
    "Porod log (log I vs log q)": ("Q", "I", True, True),
}


def transform_x(q: np.ndarray, mode: str) -> np.ndarray:
    return np.asarray(AXIS_TRANSFORMS[mode][1](q), dtype=float)


def transform_y(q: np.ndarray, I: np.ndarray, mode: str) -> np.ndarray:
    return np.asarray(Y_AXIS_TRANSFORMS[mode][1](q, I), dtype=float)


def valid_xy_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


def apply_xy_transform(q: np.ndarray, I: np.ndarray, x_mode: str, y_mode: str):
    x = transform_x(q, x_mode)
    y = transform_y(q, I, y_mode)
    mask = valid_xy_mask(x, y)
    return x[mask], y[mask], mask


def transform_label_x(mode: str) -> str:
    return AXIS_TRANSFORMS[mode][0]


def transform_label_y(mode: str) -> str:
    return Y_AXIS_TRANSFORMS[mode][0]


def inverse_transform_x(x_values, mode: str) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    if mode == "Q":
        return x
    if mode == "Q^2":
        return np.sqrt(np.clip(x, 0.0, None))
    if mode == "Q^3":
        return np.cbrt(np.clip(x, 0.0, None))
    if mode == "Q^4":
        return np.power(np.clip(x, 0.0, None), 0.25)
    if mode == "sqrt(Q)":
        return np.clip(x, 0.0, None) ** 2
    if mode == "1/Q":
        return 1.0 / np.clip(x, 1e-300, None)
    if mode == "ln(Q)":
        return np.exp(x)
    if mode == "log10(Q)":
        return np.power(10.0, x)
    if mode == "ln(Q^2)":
        return np.sqrt(np.exp(x))
    if mode == "log10(Q^2)":
        return np.sqrt(np.power(10.0, x))
    raise KeyError(mode)
