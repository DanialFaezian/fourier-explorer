import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

trapz = getattr(np, "trapezoid", None) or np.trapz


def _saw_X(f):
    w = 2 * np.pi * np.asarray(f, dtype=float)
    ws = np.where(np.abs(w) < 1e-9, 1.0, w)
    val = 2j * (np.cos(ws) / ws - np.sin(ws) / ws ** 2)
    return np.where(np.abs(w) < 1e-9, 0.0, val)


FUNCS = {
    "Rectangular pulse  rect(t)": dict(
        x=lambda t: (np.abs(t) <= 0.5).astype(float),
        X=lambda f: np.sinc(f),
        T0=4.0, tmax=2.0, fmax=6.0,
        info="X(f) = sinc(f)"),
    "Triangle pulse  tri(t)": dict(
        x=lambda t: np.clip(1 - np.abs(t), 0, None),
        X=lambda f: np.sinc(f) ** 2,
        T0=4.0, tmax=2.0, fmax=4.0,
        info="X(f) = sinc^2(f)"),
    "Gaussian  exp(-pi t^2)": dict(
        x=lambda t: np.exp(-np.pi * t ** 2),
        X=lambda f: np.exp(-np.pi * f ** 2),
        T0=6.0, tmax=6.0, fmax=3.0,
        info="X(f) = exp(-pi f^2)"),
    "Two-sided exponential  exp(-|t|)": dict(
        x=lambda t: np.exp(-np.abs(t)),
        X=lambda f: 2.0 / (1 + (2 * np.pi * f) ** 2),
        T0=12.0, tmax=30.0, fmax=2.0,
        info="X(f) = 2 / (1 + 4 pi^2 f^2)"),
    "Causal exponential  exp(-t)u(t)": dict(
        x=lambda t: np.where(t >= 0, np.exp(-np.clip(t, 0, None)), 0.0),
        X=lambda f: 1.0 / (1 + 2j * np.pi * f),
        T0=12.0, tmax=30.0, fmax=2.0,
        info="X(f) = 1 / (1 + j 2 pi f)"),
    "Sinc  sinc(t)": dict(
        x=lambda t: np.sinc(t),
        X=lambda f: (np.abs(f) <= 0.5).astype(float),
        T0=30.0, tmax=150.0, fmax=1.5,
        info="X(f) = rect(f)"),
    "Single-period sawtooth  x=t on (-1,1)": dict(
        x=lambda t: np.where(np.abs(t) < 1, t, 0.0),
        X=_saw_X,
        T0=4.0, tmax=2.0, fmax=4.0,
        info="X(f) = 2j [cos(w)/w - sin(w)/w^2],  w = 2 pi f"),
}

NAMES = list(FUNCS.keys())
NMAX = 80


def fourier_series(x_func, T, N, npts=8001):
    t = np.linspace(-T / 2, T / 2, npts)
    x = x_func(t)
    k = np.arange(0, N + 1)
    E = np.exp(-2j * np.pi * np.outer(k, t) / T)
    c = trapz(x * E, t, axis=1) / T
    a = 2 * c.real
    b = -2 * c.imag
    arg = 2 * np.pi * np.outer(k[1:], t) / T
    xN = c[0].real + (a[1:, None] * np.cos(arg) + b[1:, None] * np.sin(arg)).sum(axis=0)
    return t, x, xN, c, a, b


_num_cache = {}


def numeric_ft(name):
    if name in _num_cache:
        return _num_cache[name]
    F = FUNCS[name]
    t = np.linspace(-F["tmax"], F["tmax"], int(F["tmax"] * 200) + 1)
    x = F["x"](t)
    f = np.linspace(-F["fmax"], F["fmax"], 801)
    X = np.array([trapz(x * np.exp(-2j * np.pi * fi * t), t) for fi in f])
    _num_cache[name] = (f, X)
    return f, X


fig = plt.figure(figsize=(14, 8.5))
fig.canvas.manager.set_window_title("Fourier series approximation & Fourier transform")

ax_radio = fig.add_axes([0.01, 0.30, 0.20, 0.60])
radio = RadioButtons(ax_radio, NAMES)
ax_radio.set_title("Choose a function", fontsize=11)
for lbl in radio.labels:
    lbl.set_fontsize(8.5)

ax_t = fig.add_axes([0.29, 0.69, 0.68, 0.25])
ax_f = fig.add_axes([0.29, 0.37, 0.68, 0.25])
ax_c = fig.add_axes([0.29, 0.13, 0.68, 0.15])

ax_sN = fig.add_axes([0.29, 0.05, 0.55, 0.025])
ax_sT = fig.add_axes([0.29, 0.015, 0.55, 0.025])
sN = Slider(ax_sN, "N (harmonics)", 1, NMAX, valinit=10, valstep=1)
sT = Slider(ax_sT, "T (period)", 1, 40, valinit=FUNCS[NAMES[0]]["T0"])

state = {"name": NAMES[0]}


def update(_=None):
    name = state["name"]
    F = FUNCS[name]
    N = int(sN.val)
    T = float(sT.val)

    t, x, xN, c, a, b = fourier_series(F["x"], T, N)

    ax_t.clear()
    ax_t.plot(t, x, "b-", lw=2, label="original x(t)")
    ax_t.plot(t, xN, "r--", lw=1.5, label=f"trig. approximation (N={N})")
    mse = np.mean((x - xN) ** 2)
    ax_t.set_title(f"Time domain on [-T/2, T/2]   |   mean-square error = {mse:.2e}", fontsize=10)
    ax_t.set_xlabel("t")
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="upper right", fontsize=8)

    ax_f.clear()
    f = np.linspace(-F["fmax"], F["fmax"], 1601)
    ax_f.plot(f, np.abs(F["X"](f)), "b-", lw=2, label="exact FT (analytic formula)")
    fn, Xn = numeric_ft(name)
    ax_f.plot(fn, np.abs(Xn), "g:", lw=2.5, label="FT by numerical integration")
    k = np.arange(0, N + 1)
    fk = k / T
    mag = T * np.abs(c)
    sel = fk <= F["fmax"]
    fk_all = np.concatenate([-fk[sel][1:][::-1], fk[sel]])
    mag_all = np.concatenate([mag[sel][1:][::-1], mag[sel]])
    ax_f.stem(fk_all, mag_all, linefmt="r-", markerfmt="ro", basefmt=" ",
              label="approx. FT from series: T|c_k| at f=k/T")
    ax_f.set_title(f"Magnitude spectrum   |   {F['info']}", fontsize=10)
    ax_f.set_xlabel("f (Hz)")
    ax_f.set_xlim(-F["fmax"], F["fmax"])
    ax_f.grid(alpha=0.3)
    ax_f.legend(loc="upper right", fontsize=8)

    ax_c.clear()
    w = 0.4
    ax_c.bar(k[1:] - w / 2, a[1:], w, color="tab:orange", label="a_k (cos)")
    ax_c.bar(k[1:] + w / 2, b[1:], w, color="tab:purple", label="b_k (sin)")
    ax_c.set_title("Trigonometric coefficients", fontsize=10)
    ax_c.set_xlabel("k")
    ax_c.grid(alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=8)

    fig.canvas.draw_idle()


def on_select(label):
    state["name"] = label
    sT.set_val(FUNCS[label]["T0"])
    update()


radio.on_clicked(on_select)
sN.on_changed(update)
sT.on_changed(update)

sT.set_val(FUNCS[NAMES[0]]["T0"])
update()
plt.show()
