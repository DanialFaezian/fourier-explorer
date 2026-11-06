# Fourier Explorer

An interactive matplotlib app that approximates a signal with a trigonometric Fourier series and compares the result with its Fourier transform.

Pick one of seven signals, then use the sliders to change the number of harmonics (N) and the period (T). The window shows:

- the original signal and its approximation, with the mean-square error
- the magnitude spectrum: the exact transform, a numerical transform, and the series estimate T|c_k| at f = k/T
- the a_k and b_k coefficients

## Run it

Requires Python 3.

```bash
pip install -r requirements.txt
python fourier_explorer.py
```
