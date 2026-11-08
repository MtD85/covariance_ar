import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.signal import chirp
from scipy.linalg import solve

# -------------------- Сигналы --------------------
def make_signal(N, fs, kind):
    t = np.arange(N) / fs
    if kind == "Два тона":
        # Комплексный двухтональный сигнал с разными амплитудами
        x = 1.0*np.exp(1j*2*np.pi*0.12*fs*t) + 0.45*np.exp(1j*2*np.pi*0.28*fs*t)
    else:
        # Комплексный ЛЧМ: мгновенная частота меняется от 0.05fs до 0.40fs
        f0, f1 = 0.05*fs, 0.40*fs
        phase = 2*np.pi*(f0*t + (f1-f0)*t**2/(2*t[-1]))
        x = np.exp(1j*phase)
    return t, x

def add_complex_noise(x, snr_db, rng):
    Ps = np.mean(np.abs(x)**2)
    Pn = Ps / (10**(snr_db/10))
    noise = np.sqrt(Pn/2)*(rng.standard_normal(len(x)) +
                           1j*rng.standard_normal(len(x)))
    return x + noise

# ---------------- Ковариационный метод ----------------
def covariance_ar(x, p):
    """
    Минимизируем sum_{n=p}^{N-1} |x[n] + sum a[k]x[n-k]|^2.
    Решение МНК: (X^H X)a = -X^H y.
    """
    N = len(x)
    if p >= N-1:
        raise ValueError("Порядок p должен быть меньше N-1")


    X = np.empty((N-p, p), dtype=complex)
    y = x[p:]
    for k in range(1, p+1):
        X[:, k-1] = x[p-k:N-k]


    a, *_ = np.linalg.lstsq(X, -y, rcond=None)
    e = y + X @ a
    sigma_e2 = np.mean(np.abs(e)**2)
    return a, sigma_e2

def ar_spectrum(a, sigma_e2, fs, nfft=4096):
    f = np.linspace(0, fs, nfft, endpoint=False)
    den = np.ones(nfft, dtype=complex)
    for k, ak in enumerate(a, start=1):
        den += ak*np.exp(-1j*2*np.pi*f*k/fs)
    P = sigma_e2 / np.maximum(np.abs(den)**2, 1e-14)
    return f, 10*np.log10(P + 1e-15)

# -------------------- Интерфейс --------------------
N0 = 512
fs = 1000.0
rng = np.random.default_rng(42)
snr0 = 0
p0 = 24
kind0 = "Два тона"

fig = plt.figure(figsize=(13, 8))
fig.suptitle("Ковариационный метод оценки АР-модели", fontsize=16)

ax_time = fig.add_axes([0.08, 0.57, 0.56, 0.30])
ax_spec = fig.add_axes([0.08, 0.14, 0.56, 0.32])
ax_kind = fig.add_axes([0.72, 0.65, 0.23, 0.18])
ax_snr = fig.add_axes([0.72, 0.48, 0.22, 0.035])
ax_order = fig.add_axes([0.72, 0.38, 0.22, 0.035])
ax_n = fig.add_axes([0.72, 0.28, 0.22, 0.035])
ax_button = fig.add_axes([0.75, 0.16, 0.16, 0.06])

radio = RadioButtons(ax_kind, ("Два тона", "ЛЧМ"), active=0)
slider_snr = Slider(ax_snr, "ОСШ, дБ", -10, 40, valinit=snr0, valstep=1)
slider_p = Slider(ax_order, "Порядок p", 2, 100, valinit=p0, valstep=1)
slider_n = Slider(ax_n, "Число N", 128, 2048, valinit=N0, valstep=64)
button = Button(ax_button, "Новый шум")

def update(_=None):
    N = int(slider_n.val)
    p = min(int(slider_p.val), N-2)
    snr = slider_snr.val
    kind = radio.value_selected

    t, x = make_signal(N, fs, kind)
    xn = add_complex_noise(x, snr, rng)
    a, err = covariance_ar(xn, p)

    # Временная область
    ax_time.clear()
    ax_time.plot(t[:min(N, 250)], xn.real[:min(N, 250)], label="Re{x[n]} с шумом")
    ax_time.plot(t[:min(N, 250)], x.real[:min(N, 250)], alpha=0.8, label="Чистый сигнал")
    ax_time.set_title(f"Сигнал во времени | ОСШ = {snr:.0f} дБ")
    ax_time.set_xlabel("Время, с")
    ax_time.set_ylabel("Амплитуда")
    ax_time.grid(True)
    ax_time.legend(loc="upper right")

    # Спектр
    f, par = ar_spectrum(a, err, fs)
    ax_spec.clear()
    ax_spec.plot(f/fs, par, label="Ковариационная АР-оценка")
    ax_spec.set_xlim(0, 0.5)
    ax_spec.set_title(f"АР-спектральная оценка (p={p}, N={N})")
    ax_spec.set_xlabel("Нормированная частота f/fs")
    ax_spec.set_ylabel("Мощность, дБ")
    ax_spec.grid(True)
    ax_spec.legend()
    fig.canvas.draw_idle()

slider_snr.on_changed(update)
slider_p.on_changed(update)
slider_n.on_changed(update)
radio.on_clicked(update)
button.on_clicked(update)
update()
plt.show()
