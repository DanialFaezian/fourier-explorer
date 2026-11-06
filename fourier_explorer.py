
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fourier Explorer",
    page_icon="📈",
    layout="wide"
)

st.title("Fourier Series & Fourier Transform Explorer")

st.markdown(
    "Explore Fourier series approximations and Fourier transforms "
    "of different signals interactively."
)

trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------
# Fourier transform definitions
# --------------------------------------------------

def _saw_X(f):
    w = 2 * np.pi * np.asarray(f, dtype=float)
    ws = np.where(np.abs(w) < 1e-9, 1.0, w)

    val = 2j * (
        np.cos(ws) / ws -
        np.sin(ws) / ws**2
    )

    return np.where(np.abs(w) < 1e-9, 0.0, val)


FUNCS = {
    "Rectangular pulse  rect(t)": dict(
        x=lambda t: (np.abs(t) <= 0.5).astype(float),
        X=lambda f: np.sinc(f),
        T0=4.0,
        tmax=2.0,
        fmax=6.0,
        info="X(f) = sinc(f)"
    ),

    "Triangle pulse  tri(t)": dict(
        x=lambda t: np.clip(1 - np.abs(t), 0, None),
        X=lambda f: np.sinc(f)**2,
        T0=4.0,
        tmax=2.0,
        fmax=4.0,
        info="X(f) = sinc²(f)"
    ),

    "Gaussian  exp(-pi t²)": dict(
        x=lambda t: np.exp(-np.pi * t**2),
        X=lambda f: np.exp(-np.pi * f**2),
        T0=6.0,
        tmax=6.0,
        fmax=3.0,
        info="X(f) = exp(-πf²)"
    ),

    "Two-sided exponential  exp(-|t|)": dict(
        x=lambda t: np.exp(-np.abs(t)),
        X=lambda f: 2.0 / (1 + (2 * np.pi * f)**2),
        T0=12.0,
        tmax=30.0,
        fmax=2.0,
        info="X(f) = 2 / (1 + 4π²f²)"
    ),

    "Causal exponential  exp(-t)u(t)": dict(
        x=lambda t: np.where(
            t >= 0,
            np.exp(-np.clip(t, 0, None)),
            0.0
        ),
        X=lambda f: 1.0 / (1 + 2j * np.pi * f),
        T0=12.0,
        tmax=30.0,
        fmax=2.0,
        info="X(f) = 1 / (1 + j2πf)"
    ),

    "Sinc  sinc(t)": dict(
        x=lambda t: np.sinc(t),
        X=lambda f: (np.abs(f) <= 0.5).astype(float),
        T0=30.0,
        tmax=150.0,
        fmax=1.5,
        info="X(f) = rect(f)"
    ),

    "Single-period sawtooth  x=t on (-1,1)": dict(
        x=lambda t: np.where(np.abs(t) < 1, t, 0.0),
        X=_saw_X,
        T0=4.0,
        tmax=2.0,
        fmax=4.0,
        info="X(f) = 2j[cos(w)/w - sin(w)/w²], w = 2πf"
    )
}


NAMES = list(FUNCS.keys())
NMAX = 80


# --------------------------------------------------
# Fourier series calculation
# --------------------------------------------------

def fourier_series(x_func, T, N, npts=8001):

    t = np.linspace(-T / 2, T / 2, npts)
    x = x_func(t)

    k = np.arange(0, N + 1)

    E = np.exp(
        -2j * np.pi * np.outer(k, t) / T
    )

    c = trapz(x * E, t, axis=1) / T

    a = 2 * c.real
    b = -2 * c.imag

    arg = (
        2 * np.pi *
        np.outer(k[1:], t) / T
    )

    xN = (
        c[0].real +
        (
            a[1:, None] * np.cos(arg) +
            b[1:, None] * np.sin(arg)
        ).sum(axis=0)
    )

    return t, x, xN, c, a, b


# --------------------------------------------------
# Numerical Fourier transform
# --------------------------------------------------

@st.cache_data
def numeric_ft(name):

    F = FUNCS[name]

    t = np.linspace(
        -F["tmax"],
        F["tmax"],
        int(F["tmax"] * 200) + 1
    )

    x = F["x"](t)

    f = np.linspace(
        -F["fmax"],
        F["fmax"],
        801
    )

    X = np.array([
        trapz(
            x * np.exp(-2j * np.pi * fi * t),
            t
        )
        for fi in f
    ])

    return f, X


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

st.sidebar.header("Signal settings")

name = st.sidebar.selectbox(
    "Choose a function",
    NAMES
)

F = FUNCS[name]

N = st.sidebar.slider(
    "Number of harmonics (N)",
    min_value=1,
    max_value=NMAX,
    value=10,
    step=1
)

T = st.sidebar.slider(
    "Period (T)",
    min_value=1.0,
    max_value=40.0,
    value=float(F["T0"]),
    step=0.5
)

st.sidebar.divider()

st.sidebar.markdown("### Selected function")

st.sidebar.write(name)

st.sidebar.info(F["info"])


# --------------------------------------------------
# Calculate Fourier series
# --------------------------------------------------

t, x, xN, c, a, b = fourier_series(
    F["x"],
    T,
    N
)

mse = np.mean((x - xN)**2)


# --------------------------------------------------
# 1. Time domain
# --------------------------------------------------

st.header("1. Time Domain")

st.markdown(
    "Compare the original signal with its trigonometric "
    "Fourier series approximation."
)

fig1, ax1 = plt.subplots(figsize=(12, 4))

ax1.plot(
    t,
    x,
    "b-",
    linewidth=2,
    label="Original x(t)"
)

ax1.plot(
    t,
    xN,
    "r--",
    linewidth=1.5,
    label=f"Fourier approximation (N={N})"
)

ax1.set_title(
    f"Time Domain | Mean Square Error = {mse:.3e}"
)

ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Amplitude")

ax1.grid(alpha=0.3)
ax1.legend()

fig1.tight_layout()

st.pyplot(fig1)
plt.close(fig1)

st.metric(
    "Mean Square Error",
    f"{mse:.6e}"
)


# --------------------------------------------------
# 2. Fourier transform
# --------------------------------------------------

st.header("2. Fourier Transform")

st.markdown(
    "Compare the analytical Fourier transform, numerical "
    "integration, and Fourier series spectrum."
)

f = np.linspace(
    -F["fmax"],
    F["fmax"],
    1601
)

fig2, ax2 = plt.subplots(figsize=(12, 4))

# Exact Fourier transform
ax2.plot(
    f,
    np.abs(F["X"](f)),
    "b-",
    linewidth=2,
    label="Exact FT (analytical)"
)

# Numerical Fourier transform
fn, Xn = numeric_ft(name)

ax2.plot(
    fn,
    np.abs(Xn),
    "g:",
    linewidth=2.5,
    label="Numerical FT"
)

# Fourier series spectrum
k = np.arange(0, N + 1)

fk = k / T
mag = T * np.abs(c)

sel = fk <= F["fmax"]

fk_all = np.concatenate([
    -fk[sel][1:][::-1],
    fk[sel]
])

mag_all = np.concatenate([
    mag[sel][1:][::-1],
    mag[sel]
])

ax2.stem(
    fk_all,
    mag_all,
    linefmt="r-",
    markerfmt="ro",
    basefmt=" ",
    label="Fourier series spectrum: T|cₖ|"
)

ax2.set_title(
    f"Magnitude Spectrum | {F['info']}"
)

ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("|X(f)|")

ax2.set_xlim(
    -F["fmax"],
    F["fmax"]
)

ax2.grid(alpha=0.3)
ax2.legend()

fig2.tight_layout()

st.pyplot(fig2)
plt.close(fig2)


# --------------------------------------------------
# 3. Trigonometric coefficients
# --------------------------------------------------

st.header("3. Trigonometric Fourier Coefficients")

st.markdown(
    "The coefficients of the trigonometric Fourier series:"
)

col1, col2 = st.columns(2)

with col1:
    st.latex(
        r"x_N(t)=a_0/2+"
        r"\sum_{k=1}^{N}"
        r"[a_k\cos(2\pi kt/T)+"
        r"b_k\sin(2\pi kt/T)]"
    )

with col2:
    st.latex(
        r"c_k=\frac{1}{T}"
        r"\int_{-T/2}^{T/2}"
        r"x(t)e^{-j2\pi kt/T}\,dt"
    )

fig3, ax3 = plt.subplots(figsize=(12, 4))

ax3.bar(
    k[1:] - 0.2,
    a[1:],
    0.4,
    color="tab:orange",
    label="aₖ (cosine)"
)

ax3.bar(
    k[1:] + 0.2,
    b[1:],
    0.4,
    color="tab:purple",
    label="bₖ (sine)"
)

ax3.set_title("Trigonometric Coefficients")
ax3.set_xlabel("Harmonic number (k)")
ax3.set_ylabel("Coefficient value")

ax3.grid(alpha=0.3)
ax3.legend()

fig3.tight_layout()

st.pyplot(fig3)
plt.close(fig3)


# --------------------------------------------------
# 4. Coefficient table
# --------------------------------------------------

st.header("4. Coefficient Values")

st.dataframe(
    {
        "k": k[1:],
        "a_k (cosine)": a[1:],
        "b_k (sine)": b[1:],
        "c_k (real)": c[1:].real,
        "c_k (imaginary)": c[1:].imag,
        "Frequency (k/T)": fk[1:]
    },
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Fourier Explorer | Interactive Fourier Series "
    "and Fourier Transform Visualization"
)
