# -*- coding: utf-8 -*-
"""Adaptive_Equalizer.ipynb
Author: Yashaswi Karthik Reddy Danda
Contact: +1 682 262 6288
Email: kd3400@rit.edu
Version: 7
Implemented through Google Colab
Original file is located at
    https://colab.research.google.com/drive/1WNui_WBxl_p5Bq-ZfEL3LydbDpOjbPhR
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# ─── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ─── Parameters (user configurable) ──────────────────────────────────────────
N_SYMBOLS    = 8000       # total BPSK symbols
SNR_DB       = 20         # signal to noise ratio in dB
CHANNEL_VARY = True       # whether channel changes over time (non-stationary)
VARY_EVERY   = 500        # channel changes every N symbols

# LMS parameters
N_TAPS       = 7          # number of equalizer taps
MU           = 0.1     # LMS step size (learning rate) — reduced for lower steady-state jitter
DELAY        = N_TAPS // 2  # decision delay (center tap alignment)

# ─── BPSK Signal Generation ───────────────────────────────────────────────────
def generate_bpsk(n_symbols):
    bits = np.random.randint(0, 2, n_symbols)
    symbols = 2 * bits - 1   # 0 → -1, 1 → +1
    return bits, symbols.astype(np.float64)

# ─── Multipath Channel Model ──────────────────────────────────────────────────
def get_channel(profile='pedestrian_a'):
    profiles = {
        'pedestrian_a': np.array([1.0, 0.5, 0.2]),
        'vehicular_a':  np.array([1.0, 0.8, 0.5, 0.3, 0.1]),
        'flat':         np.array([1.0]),
    }
    h = profiles.get(profile, profiles['pedestrian_a'])
    return h / np.linalg.norm(h)

def apply_channel(symbols, h):
    return lfilter(h, [1.0], symbols)

def add_noise(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

def generate_dataset(n_symbols, snr_db, channel_vary=True, vary_every=500):
    bits, tx = generate_bpsk(n_symbols)
    rx = np.zeros(n_symbols)

    if channel_vary:
        profiles = ['pedestrian_a', 'vehicular_a']
        for i in range(0, n_symbols, vary_every):
            end = min(i + vary_every, n_symbols)
            profile = profiles[(i // vary_every) % 2]
            h = get_channel(profile)
            segment = tx[i:end]
            rx[i:end] = apply_channel(segment, h)[:end - i]
    else:
        h = get_channel('pedestrian_a')
        rx = apply_channel(tx, h)

    rx_noisy = add_noise(rx, snr_db)
    return bits, tx, rx_noisy

# ─── LMS Equalizer ────────────────────────────────────────────────────────────
def lms_equalizer(rx, tx_ref, n_taps, mu, delay):
    """
    LMS adaptive equalizer with normalized step size (NLMS).

    Args:
        rx      : received (distorted + noisy) signal
        tx_ref  : known transmitted symbols used as training reference
        n_taps  : number of FIR equalizer taps
        mu      : step size (learning rate). Smaller = more stable, slower convergence.
        delay   : decision delay — aligns the equalizer output with the reference.
                  Typically set to n_taps // 2 (center tap).

    Returns:
        eq_out  : equalized signal (aligned)
        weights : final tap weight vector  [n_taps]
        mse     : per-symbol MSE history   [N]
        weight_history : tap evolution     [N x n_taps]

    Algorithm (per sample n):
        x(n)   = [rx(n), rx(n-1), ..., rx(n-L+1)]   input buffer
        y(n)   = w(n)^T · x(n)                        filter output
        d(n)   = tx_ref(n - delay)                    desired (delayed reference)
        e(n)   = d(n) - y(n)                          error
        μ_eff  = μ / (ε + ||x||²)                     normalized step (NLMS)
        w(n+1) = w(n) + μ_eff * e(n) * x(n)          weight update
    """
    N = len(rx)
    w = np.zeros(n_taps)           # initialize taps to zero
    eq_out = np.zeros(N)
    mse = np.zeros(N)
    weight_history = np.zeros((N, n_taps))

    for n in range(N):
        # Build input vector: [rx(n), rx(n-1), ..., rx(n-L+1)]
        start = max(0, n - n_taps + 1)
        x = rx[start:n + 1][::-1]           # most-recent first
        x = np.pad(x, (0, n_taps - len(x))) # zero-pad at startup

        # Filter output
        y = np.dot(w, x)
        eq_out[n] = y

        # Desired symbol (with decision delay)
        d_idx = n - delay
        d = tx_ref[d_idx] if d_idx >= 0 else 0.0

        # Error and NLMS weight update: μ_eff = μ / (ε + ||x||²)
        e = d - y
        mu_eff = mu / (1e-6 + np.dot(x, x))
        w = w + mu_eff * e * x

        mse[n] = e ** 2
        weight_history[n] = w

    return eq_out, w, mse, weight_history

# ─── BER Calculation ─────────────────────────────────────────────────────────
def compute_ber(bits_tx, eq_out, delay):
    """Hard decision + BER, accounting for equalizer delay."""
    decisions = (eq_out[delay:] > 0).astype(int)
    ref_bits   = bits_tx[:len(decisions)]
    errors     = np.sum(decisions != ref_bits)
    return errors / len(ref_bits)

# ─── BER vs SNR Sweep ────────────────────────────────────────────────────────
def ber_vs_snr(snr_range, n_symbols=10000, n_taps=11, mu=0.1, delay=None,
               n_trials=8):
    """
    Monte Carlo BER sweep over Vehicular A channel (severe ISI, 5 taps).

    Alignment note:
        lms_equalizer internally uses d[n] = tx[n - delay] as the desired
        symbol, so eq_out[n] is an estimate of tx[n - delay].
        Correct comparison: eq_out[n]  <-->  bits[n - delay]
        i.e. for measurement window starting at index W:
             eq_dec  = sign(eq_out[W:])
             ref_eq  = bits[W - delay : N - delay]   (shift reference back)
    """
    from scipy.special import erfc
    if delay is None:
        delay = n_taps // 2

    # Convergence guard: skip first 20% of symbols so taps are settled
    conv_guard = int(0.20 * n_symbols)
    MIN_BER    = 1e-5

    # Vehicular A — unnormalized for strong ISI
    h_veh = np.array([1.0, 0.8, 0.5, 0.3, 0.1])

    ber_raw, ber_eq = [], []
    print(f"\n  {'SNR':>4}  {'BER_raw':>10}  {'BER_eq':>10}")
    print(f"  {'─'*30}")

    for snr_db in snr_range:
        raw_errors, raw_total = 0, 0
        eq_errors,  eq_total  = 0, 0

        for _ in range(n_trials):
            bits, tx = generate_bpsk(n_symbols)
            rx_clean = apply_channel(tx, h_veh)
            rx       = add_noise(rx_clean, snr_db)

            # ── Raw BER: simple threshold on received signal, post guard ──────
            raw_dec    = (rx[conv_guard:] > 0).astype(int)
            ref_raw    = bits[conv_guard: conv_guard + len(raw_dec)]
            n_raw      = min(len(raw_dec), len(ref_raw))
            raw_errors += np.sum(raw_dec[:n_raw] != ref_raw[:n_raw])
            raw_total  += n_raw

            # ── NLMS BER: correct delay alignment ────────────────────────────
            # eq_out[n] estimates tx[n - delay]
            # measurement window in eq_out: [conv_guard .. N-1]
            # corresponding tx indices:     [conv_guard-delay .. N-1-delay]
            # so reference bits:            bits[conv_guard-delay : N-delay]
            eq_out, _, _, _ = lms_equalizer(rx, tx, n_taps, mu, delay)

            eq_start   = conv_guard
            ref_start  = conv_guard - delay          # shift reference by delay
            ref_end    = n_symbols - delay

            if ref_start < 0:                        # safety: guard too small
                eq_start  -= ref_start
                ref_start  = 0

            eq_dec   = (eq_out[eq_start:] > 0).astype(int)
            ref_eq   = bits[ref_start: ref_end]
            n_valid  = min(len(eq_dec), len(ref_eq))

            eq_errors += np.sum(eq_dec[:n_valid] != ref_eq[:n_valid])
            eq_total  += n_valid

        b_raw = raw_errors / raw_total if raw_total > 0 else MIN_BER
        b_eq  = eq_errors  / eq_total  if eq_total  > 0 else MIN_BER
        ber_raw.append(max(b_raw, MIN_BER))
        ber_eq.append(max(b_eq,  MIN_BER))
        print(f"  {snr_db:>4}  {b_raw:>10.2e}  {b_eq:>10.2e}")

    snr_lin    = 10 ** (np.array(snr_range) / 10)
    ber_theory = 0.5 * erfc(np.sqrt(snr_lin))

    return ber_raw, ber_eq, ber_theory

# ─── Eye Diagram ─────────────────────────────────────────────────────────────
def plot_eye_diagram(ax, signal, sps=2, n_traces=200, title='Eye Diagram', color='#f87171'):
    """Overlay signal traces of length 2*sps to form an eye diagram."""
    seg_len = 2 * sps
    for i in range(n_traces):
        start = i * sps
        end = start + seg_len
        if end > len(signal):
            break
        ax.plot(signal[start:end], color=color, alpha=0.15, linewidth=0.6)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')

# ─── Run Everything ───────────────────────────────────────────────────────────
bits, tx, rx = generate_dataset(N_SYMBOLS, SNR_DB, CHANNEL_VARY, VARY_EVERY)

eq_out, final_weights, mse_history, weight_history = lms_equalizer(
    rx, tx, N_TAPS, MU, DELAY
)

ber = compute_ber(bits, eq_out, DELAY)

# Smoothed MSE (running average over 50 samples)
smooth_mse = np.convolve(mse_history, np.ones(50)/50, mode='valid')

print(f"{'─'*50}")
print(f"  N_SYMBOLS : {N_SYMBOLS}")
print(f"  SNR       : {SNR_DB} dB")
print(f"  N_TAPS    : {N_TAPS}   MU : {MU}   DELAY : {DELAY}")
print(f"  Final BER : {ber:.4f}  ({ber*100:.2f}%)")
print(f"  Final MSE : {np.mean(mse_history[-500:]):.5f}  (last 500 samples)")
print(f"  BER sweep : Vehicular A channel, 8 Monte Carlo trials per SNR point")
print(f"{'─'*50}")

# ─── Fixed-channel pass for clean eye diagrams ───────────────────────────────
# Operational eye: SNR=20dB, same config as main run
_, tx_eye, rx_eye = generate_dataset(N_SYMBOLS, SNR_DB, channel_vary=False)
eq_eye, _, _, _ = lms_equalizer(rx_eye, tx_eye, N_TAPS, MU, DELAY)

# Best-case eye: SNR=30dB, N_TAPS=9 — shows fully-open reference eye
DELAY_EYE2 = 9 // 2
_, tx_eye2, rx_eye2 = generate_dataset(N_SYMBOLS, 30, channel_vary=False)
eq_eye2, _, _, _ = lms_equalizer(rx_eye2, tx_eye2, 9, MU, DELAY_EYE2)

SNR_RANGE = np.arange(0, 22, 1)
ber_raw, ber_eq, ber_theory = ber_vs_snr(SNR_RANGE, n_taps=11, mu=MU)

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor('#080b10')

ax_style = dict(facecolor='#0e1319',
                tick_params=dict(colors='gray'),
                spine_color='#2a3441')

def style_ax(ax):
    ax.set_facecolor('#0e1319')
    ax.tick_params(colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')
    ax.title.set_color('white')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#2a3441')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── 1. Received vs Equalized signal (first 200 samples) ──────────────────────
ax1 = fig.add_subplot(5, 2, 1)
ax1.plot(rx[:200], color='#f87171', linewidth=0.8, label='Received')
ax1.plot(eq_out[DELAY:200+DELAY], color='#4ade80', linewidth=0.8, label='Equalized')
ax1.set_title('Received vs Equalized (first 200 symbols)', fontweight='bold')
ax1.set_ylabel('Amplitude')
ax1.legend(fontsize=8, facecolor='#0e1319', labelcolor='gray')
for k in range(0, N_SYMBOLS, VARY_EVERY):
    if k < 200:
        ax1.axvline(k, color='yellow', alpha=0.4, linestyle='--', linewidth=0.7)
style_ax(ax1)

# ── 2. MSE convergence ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(5, 2, 2)
ax2.plot(smooth_mse, color='#a3e635', linewidth=0.9)
ax2.set_title('NLMS MSE Convergence (smoothed)', fontweight='bold')
ax2.set_ylabel('MSE')
ax2.set_xlabel('Symbol index')
for k in range(0, N_SYMBOLS, VARY_EVERY):
    ax2.axvline(k, color='yellow', alpha=0.3, linestyle='--', linewidth=0.7)
style_ax(ax2)

# ── 3. Final tap weights ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(5, 2, 3)
markerline, stemlines, _ = ax3.stem(final_weights, linefmt='C0-', markerfmt='C0o', basefmt=' ')
plt.setp(stemlines, color='#60a5fa')
plt.setp(markerline, color='#60a5fa')
ax3.set_title(f'Final Tap Weights (N={N_TAPS})', fontweight='bold')
ax3.set_ylabel('Weight')
ax3.set_xlabel('Tap index')
style_ax(ax3)

# ── 4. Tap weight evolution ───────────────────────────────────────────────────
ax4 = fig.add_subplot(5, 2, 4)
colors_tap = plt.cm.plasma(np.linspace(0.2, 0.9, N_TAPS))
for i in range(N_TAPS):
    ax4.plot(weight_history[::10, i], color=colors_tap[i], linewidth=0.7,
             label=f'w{i}' if i < 4 else None)
ax4.set_title('Tap Weight Evolution', fontweight='bold')
ax4.set_ylabel('Weight')
ax4.set_xlabel('Symbol (×10)')
ax4.legend(fontsize=7, facecolor='#0e1319', labelcolor='gray', ncol=2)
style_ax(ax4)

# ── 5. BER vs SNR ─────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(5, 2, 5)
ax5.semilogy(SNR_RANGE, ber_raw,    color='#f87171', marker='o', linewidth=1.2,
             markersize=4, label='No equalizer')
ax5.semilogy(SNR_RANGE, ber_eq,     color='#4ade80', marker='s', linewidth=1.2,
             markersize=4, label='NLMS equalized')
# Clip theory curve to same floor as data so it doesn't collapse the y-axis
ber_theory_clipped = np.clip(ber_theory, 1e-5, 1.0)
ax5.semilogy(SNR_RANGE, ber_theory_clipped, color='#a3e635', linestyle='--', linewidth=1.2,
             label='AWGN theory (clipped)')
ax5.set_ylim([1e-5, 1.0])          # fix y-axis to data range — prevents theory curve collapsing view
ax5.set_title('BER vs SNR — Vehicular A channel (Monte Carlo, 8 trials)', fontweight='bold')
ax5.set_ylabel('BER')
ax5.set_xlabel('SNR (dB)')
ax5.legend(fontsize=8, facecolor='#0e1319', labelcolor='gray')
ax5.grid(True, alpha=0.15, color='gray')
style_ax(ax5)

# ── 6. Spectrum: received vs equalized ────────────────────────────────────────
ax6 = fig.add_subplot(5, 2, 6)
ax6.magnitude_spectrum(rx,              Fs=1, color='#f87171', linewidth=0.8, label='Received')
ax6.magnitude_spectrum(eq_out[DELAY:],  Fs=1, color='#4ade80', linewidth=0.8, label='Equalized')
ax6.set_title('Spectrum: Received vs Equalized', fontweight='bold')
ax6.legend(fontsize=8, facecolor='#0e1319', labelcolor='gray')
style_ax(ax6)

# ── 7. Eye diagram — before equalizer (operational, SNR=20dB) ─────────────────
ax7 = fig.add_subplot(5, 2, 7)
plot_eye_diagram(ax7, rx_eye, sps=2, n_traces=300,
                 title='Eye Diagram — Before Equalizer (20dB)', color='#f87171')
style_ax(ax7)

# ── 8. Eye diagram — after equalizer (operational, SNR=20dB) ─────────────────
ax8 = fig.add_subplot(5, 2, 8)
plot_eye_diagram(ax8, eq_eye[DELAY:], sps=2, n_traces=300,
                 title='Eye Diagram — After NLMS (20dB, operational)', color='#4ade80')
style_ax(ax8)

# ── 9. Best-case eye — before (SNR=30dB, 9 taps) ─────────────────────────────
ax9 = fig.add_subplot(5, 2, 9)
plot_eye_diagram(ax9, rx_eye2, sps=2, n_traces=300,
                 title='Eye Diagram — Before Equalizer (30dB, reference)', color='#fb923c')
style_ax(ax9)

# ── 10. Best-case eye — after (SNR=30dB, 9 taps) ─────────────────────────────
ax10 = fig.add_subplot(5, 2, 10)
plot_eye_diagram(ax10, eq_eye2[DELAY_EYE2:], sps=2, n_traces=300,
                 title='Eye Diagram — After NLMS (30dB, 9 taps, best-case)', color='#34d399')
style_ax(ax10)

plt.suptitle(f'NLMS Equalizer — BPSK over Multipath Channel  |  SNR={SNR_DB}dB  |  Taps={N_TAPS}  |  μ={MU}',
             color='white', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('lms_equalizer_results.png', dpi=150, bbox_inches='tight',
            facecolor='#080b10')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — μ OPTIMIZATION TECHNIQUE COMPARISON
# Compares 4 equalizer update rules on BER vs SNR over Vehicular A channel:
#   1. NLMS (baseline)
#   2. Momentum SGD
#   3. RMSProp
#   4. Mini-batch LMS
# ══════════════════════════════════════════════════════════════════════════════

# ─── Optimizer 1: NLMS (baseline, already defined above) ─────────────────────
# Uses lms_equalizer() with NLMS update: w += (μ / (ε + ||x||²)) * e * x

# ─── Optimizer 2: Momentum SGD equalizer ─────────────────────────────────────
def momentum_equalizer(rx, tx_ref, n_taps, mu, delay, beta=0.9):
    """
    Momentum SGD equalizer (Heavy Ball method).

    Adds a velocity accumulator to the NLMS update:
        v(n+1) = beta * v(n) + mu_eff * e(n) * x(n)
        w(n+1) = w(n) + v(n+1)

    beta  : momentum coefficient (0.9 typical).
            High beta → more inertia, smoother convergence, slower adaptation.
            Low beta  → closer to plain NLMS.

    Why useful for wireless: reduces tap oscillation at channel switches by
    smoothing out the noisy gradient estimates symbol-by-symbol.
    """
    N = len(rx)
    w = np.zeros(n_taps)
    v = np.zeros(n_taps)        # velocity accumulator
    eq_out = np.zeros(N)
    mse    = np.zeros(N)
    weight_history = np.zeros((N, n_taps))

    for n in range(N):
        start = max(0, n - n_taps + 1)
        x = rx[start:n+1][::-1]
        x = np.pad(x, (0, n_taps - len(x)))

        y = np.dot(w, x)
        eq_out[n] = y

        d_idx = n - delay
        d = tx_ref[d_idx] if d_idx >= 0 else 0.0
        e = d - y

        mu_eff = mu / (1e-6 + np.dot(x, x))   # normalized step
        v = beta * v + mu_eff * e * x          # momentum update
        w = w + v

        mse[n] = e ** 2
        weight_history[n] = w

    return eq_out, w, mse, weight_history


# ─── Optimizer 3: RMSProp equalizer ──────────────────────────────────────────
def rmsprop_equalizer(rx, tx_ref, n_taps, mu, delay, beta=0.99, eps=1e-6):
    """
    RMSProp adaptive equalizer.

    Per-tap exponential moving average of squared gradients:
        G_k(n+1) = beta * G_k(n) + (1 - beta) * g_k(n)²
        w_k(n+1) = w_k(n) + (mu / sqrt(G_k + eps)) * g_k(n)

    where g_k(n) = e(n) * x_k(n)  (the per-tap gradient).

    beta  : forgetting factor (0.99 typical). Controls how fast old gradient
            history decays. High beta → long memory, slow adaptation.
            Low beta  → short memory, fast adaptation but noisy.
    eps   : prevents division by zero on inactive taps.

    Why best for non-stationary wireless: exponential forgetting naturally
    'forgets' the old channel and re-scales per-tap step sizes to current
    gradient magnitudes after each channel switch.
    """
    N = len(rx)
    w = np.zeros(n_taps)
    G = np.ones(n_taps) * eps   # init to eps, not zero, for stable first step
    eq_out = np.zeros(N)
    mse    = np.zeros(N)
    weight_history = np.zeros((N, n_taps))

    for n in range(N):
        start = max(0, n - n_taps + 1)
        x = rx[start:n+1][::-1]
        x = np.pad(x, (0, n_taps - len(x)))

        y = np.dot(w, x)
        eq_out[n] = y

        d_idx = n - delay
        d = tx_ref[d_idx] if d_idx >= 0 else 0.0
        e = d - y

        g = e * x                                      # per-tap gradient
        G = beta * G + (1 - beta) * g ** 2            # EMA of squared gradient
        w = w + (mu / (np.sqrt(G) + eps)) * g         # per-tap adaptive step

        mse[n] = e ** 2
        weight_history[n] = w

    return eq_out, w, mse, weight_history


# ─── Optimizer 4: Mini-batch LMS equalizer ───────────────────────────────────
def minibatch_equalizer(rx, tx_ref, n_taps, mu, delay, batch_size=16):
    """
    Mini-batch LMS equalizer.

    Accumulates gradients over a block of B symbols before updating taps:
        Δw = (mu / B) * Σ_{i=0}^{B-1} e(n+i) * x(n+i)
        w(n+B) = w(n) + Δw

    batch_size : block length B. Typical values: 8–32.
                 Larger B → more stable update (lower gradient variance),
                 slower adaptation. Smaller B → closer to sample-by-sample.

    Why relevant for wireless: directly models pilot-based training in
    LTE/5G where the receiver processes a known pilot block (e.g. DMRS,
    16–32 subcarriers) before making data decisions. The batch update
    corresponds to processing one OFDM pilot symbol at a time.
    """
    N = len(rx)
    w = np.zeros(n_taps)
    eq_out = np.zeros(N)
    mse    = np.zeros(N)
    weight_history = np.zeros((N, n_taps))

    n = 0
    while n < N:
        batch_end = min(n + batch_size, N)
        grad_accum = np.zeros(n_taps)
        batch_mse  = 0.0

        for i in range(n, batch_end):
            start = max(0, i - n_taps + 1)
            x = rx[start:i+1][::-1]
            x = np.pad(x, (0, n_taps - len(x)))

            y = np.dot(w, x)
            eq_out[i] = y

            d_idx = i - delay
            d = tx_ref[d_idx] if d_idx >= 0 else 0.0
            e = d - y

            grad_accum += e * x
            batch_mse  += e ** 2
            mse[i]      = e ** 2
            weight_history[i] = w   # record pre-update weights for this sample

        # Single weight update per batch
        B_actual = batch_end - n
        w = w + (mu / B_actual) * grad_accum

        n = batch_end

    return eq_out, w, mse, weight_history


# ─── BER sweep for a given equalizer function ────────────────────────────────
def ber_sweep_optimizer(equalizer_fn, snr_range, n_symbols=10000, n_taps=11,
                        mu=0.1, delay=None, n_trials=8, **eq_kwargs):
    """
    Generic Monte Carlo BER sweep that accepts any equalizer function.
    equalizer_fn must return (eq_out, weights, mse, weight_history).
    Extra keyword args (beta, batch_size, etc.) are forwarded to equalizer_fn.
    """
    from scipy.special import erfc
    if delay is None:
        delay = n_taps // 2

    conv_guard = int(0.20 * n_symbols)
    MIN_BER    = 1e-5
    h_veh      = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
    ber_eq     = []

    for snr_db in snr_range:
        eq_errors, eq_total = 0, 0

        for _ in range(n_trials):
            bits, tx  = generate_bpsk(n_symbols)
            rx_clean  = apply_channel(tx, h_veh)
            rx        = add_noise(rx_clean, snr_db)

            eq_out, _, _, _ = equalizer_fn(rx, tx, n_taps, mu, delay,
                                           **eq_kwargs)

            eq_start  = conv_guard
            ref_start = conv_guard - delay
            ref_end   = n_symbols - delay
            if ref_start < 0:
                eq_start  -= ref_start
                ref_start  = 0

            eq_dec  = (eq_out[eq_start:] > 0).astype(int)
            ref_eq  = bits[ref_start: ref_end]
            n_valid = min(len(eq_dec), len(ref_eq))

            eq_errors += np.sum(eq_dec[:n_valid] != ref_eq[:n_valid])
            eq_total  += n_valid

        ber_eq.append(max(eq_errors / eq_total if eq_total > 0 else MIN_BER,
                          MIN_BER))

    return np.array(ber_eq)


# ─── Also compute raw (no equalizer) BER once ────────────────────────────────
def ber_sweep_raw(snr_range, n_symbols=10000, n_trials=8):
    conv_guard = int(0.20 * n_symbols)
    MIN_BER    = 1e-5
    h_veh      = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
    ber_raw    = []

    for snr_db in snr_range:
        raw_errors, raw_total = 0, 0
        for _ in range(n_trials):
            bits, tx  = generate_bpsk(n_symbols)
            rx_clean  = apply_channel(tx, h_veh)
            rx        = add_noise(rx_clean, snr_db)
            raw_dec   = (rx[conv_guard:] > 0).astype(int)
            ref       = bits[conv_guard: conv_guard + len(raw_dec)]
            n_r       = min(len(raw_dec), len(ref))
            raw_errors += np.sum(raw_dec[:n_r] != ref[:n_r])
            raw_total  += n_r
        ber_raw.append(max(raw_errors / raw_total, MIN_BER))

    return np.array(ber_raw)


# ─── Run all sweeps ───────────────────────────────────────────────────────────
OPT_SNR_RANGE = np.arange(0, 22, 1)
OPT_N_TAPS    = 11
OPT_N_SYMBOLS = 10000
OPT_N_TRIALS  = 8

print("\n Running optimizer comparison BER sweeps...")
print(" (4 methods × 22 SNR points × 8 trials = 704 equalizer runs)\n")

print(" [1/5] No equalizer (raw)...")
ber_raw_opt = ber_sweep_raw(OPT_SNR_RANGE, OPT_N_SYMBOLS, OPT_N_TRIALS)

print(" [2/5] NLMS baseline...")
ber_nlms = ber_sweep_optimizer(
    lms_equalizer, OPT_SNR_RANGE,
    n_symbols=OPT_N_SYMBOLS, n_taps=OPT_N_TAPS, mu=0.1,
    n_trials=OPT_N_TRIALS
)

print(" [3/5] Momentum SGD (β=0.9)...")
ber_momentum = ber_sweep_optimizer(
    momentum_equalizer, OPT_SNR_RANGE,
    n_symbols=OPT_N_SYMBOLS, n_taps=OPT_N_TAPS, mu=0.1,
    n_trials=OPT_N_TRIALS, beta=0.9
)

print(" [4/5] RMSProp (β=0.99)...")
ber_rmsprop = ber_sweep_optimizer(
    rmsprop_equalizer, OPT_SNR_RANGE,
    n_symbols=OPT_N_SYMBOLS, n_taps=OPT_N_TAPS, mu=0.005,
    n_trials=OPT_N_TRIALS, beta=0.99
)

print(" [5/5] Mini-batch LMS (B=16)...")
ber_minibatch = ber_sweep_optimizer(
    minibatch_equalizer, OPT_SNR_RANGE,
    n_symbols=OPT_N_SYMBOLS, n_taps=OPT_N_TAPS, mu=0.1,
    n_trials=OPT_N_TRIALS, batch_size=16
)

# ─── Print comparison table ───────────────────────────────────────────────────
print(f"\n  {'SNR':>4}  {'Raw':>9}  {'NLMS':>9}  {'Momentum':>9}  {'RMSProp':>9}  {'MiniBatch':>9}")
print(f"  {'─'*57}")
for i, snr in enumerate(OPT_SNR_RANGE):
    print(f"  {snr:>4}  {ber_raw_opt[i]:>9.2e}  {ber_nlms[i]:>9.2e}  "
          f"{ber_momentum[i]:>9.2e}  {ber_rmsprop[i]:>9.2e}  {ber_minibatch[i]:>9.2e}")

# ─── AWGN theory curve for optimizer plot ────────────────────────────────────
from scipy.special import erfc as _erfc
snr_lin_opt    = 10 ** (OPT_SNR_RANGE / 10)
ber_theory_opt = np.clip(0.5 * _erfc(np.sqrt(snr_lin_opt)), 1e-5, 1.0)

# ─── Plot BER comparison ──────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('#080b10')
ax.set_facecolor('#0e1319')

methods = [
    (ber_raw_opt,    '#94a3b8', 'o', '--', 'No equalizer (raw)'),
    (ber_nlms,       '#f87171', 's', '-',  'NLMS (baseline)'),
    (ber_momentum,   '#60a5fa', '^', '-',  'Momentum SGD  β=0.9'),
    (ber_rmsprop,    '#a3e635', 'D', '-',  'RMSProp  β=0.99'),
    (ber_minibatch,  '#f59e0b', 'P', '-',  'Mini-batch LMS  B=16'),
]

for ber_vals, color, marker, ls, label in methods:
    ax.semilogy(OPT_SNR_RANGE, ber_vals, color=color, marker=marker,
                linestyle=ls, linewidth=1.4, markersize=5, label=label)

# AWGN theory reference line
ax.semilogy(OPT_SNR_RANGE, ber_theory_opt, color='white', linestyle=':',
            linewidth=1.4, label='AWGN theory (reference)')

ax.set_ylim([1e-5, 1.0])
ax.set_xlim([0, 21])
ax.set_title('BER vs SNR — μ Optimizer Comparison\nVehicular A channel  |  11 taps  |  Monte Carlo 8 trials',
             fontweight='bold', color='white', fontsize=11)
ax.set_ylabel('BER', color='gray')
ax.set_xlabel('SNR (dB)', color='gray')
ax.tick_params(colors='gray')
ax.legend(fontsize=9, facecolor='#0e1319', labelcolor='gray',
          loc='lower left', framealpha=0.8)
ax.grid(True, alpha=0.15, color='gray')
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#2a3441')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#080b10')
plt.show()
print("\nDone. Saved as optimizer_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONVERGENCE SPEED & COMPLEXITY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def measure_convergence_speed(equalizer_fn, rx, tx, n_taps, mu, delay,
                               switch_points, window=50, threshold_ratio=2.0,
                               **eq_kwargs):
    """
    Measures symbols needed to re-converge after each channel switch.
    Convergence declared when smoothed MSE drops below threshold_ratio x steady-state MSE.
    """
    _, _, mse, _ = equalizer_fn(rx, tx, n_taps, mu, delay, **eq_kwargs)
    smooth  = np.convolve(mse, np.ones(window) / window, mode='same')
    steady  = np.mean(smooth[int(0.85 * len(smooth)):])
    target  = threshold_ratio * steady

    conv_symbols = []
    for sp in switch_points:
        search = smooth[sp:]
        below  = np.where(search < target)[0]
        conv_symbols.append(below[0] if len(below) > 0 else len(search))

    return conv_symbols, steady


# Generate a single non-stationary signal at SNR=20dB for convergence analysis
bits_c, tx_c, rx_c = generate_dataset(8000, 20, channel_vary=True, vary_every=500)
switch_pts = list(range(500, 8000, 500))

optimizers_conv = {
    'NLMS':           (lms_equalizer,       dict(mu=0.1)),
    'Momentum β=0.9': (momentum_equalizer,  dict(mu=0.1,   beta=0.9)),
    'RMSProp β=0.99': (rmsprop_equalizer,   dict(mu=0.005, beta=0.99)),
    'Mini-batch B=16':(minibatch_equalizer, dict(mu=0.1,   batch_size=16)),
}

conv_results  = {}
steady_mses   = {}
print("\n─── Convergence Speed after Channel Switch ───────────────────────────")
print(f"  {'Optimizer':<22} {'Mean symbols':>14} {'Std':>8} {'Steady MSE':>12}")
print(f"  {'─'*58}")

for name, (fn, kwargs) in optimizers_conv.items():
    mu_val  = kwargs.get('mu', 0.1)
    extras  = {k: v for k, v in kwargs.items() if k != 'mu'}
    speeds, steady = measure_convergence_speed(
        fn, rx_c, tx_c, OPT_N_TAPS, mu_val, OPT_N_TAPS // 2,
        switch_pts, **extras
    )
    conv_results[name] = speeds
    steady_mses[name]  = steady
    print(f"  {name:<22} {np.mean(speeds):>14.1f} {np.std(speeds):>8.1f} {steady:>12.6f}")


# ─── Complexity table ─────────────────────────────────────────────────────────
L = OPT_N_TAPS
B = 16
print(f"\n─── Computational Complexity per Symbol Update  (L={L} taps, B={B}) ───")
print(f"  {'Optimizer':<22} {'Multiplies':>13} {'Additions':>12} {'Extra memory':>14} {'Order':>8}")
print(f"  {'─'*73}")
complexity_rows = [
    ('NLMS',            2*L+1,        L,       L,      'O(L)'),
    ('Momentum β=0.9',  2*L+2,        2*L,     2*L,    'O(L)'),
    ('RMSProp β=0.99',  4*L+1,        3*L,     2*L,    'O(L)'),
    (f'Mini-batch B={B}', 2*B*L,      B*L,     L,      'O(BL)'),
    ('RLS [reference]', 2*L**2+3*L,   L**2+L,  L**2,   'O(L²)'),
]
for row in complexity_rows:
    print(f"  {row[0]:<22} {row[1]:>13} {row[2]:>12} {row[3]:>14} {row[4]:>8}")
print(f"\n  RLS shown as reference only — not run here.")
print(f"  O(L²) memory/compute cost makes RLS impractical for large L or")
print(f"  real-time embedded equalizers, which is why O(L) methods dominate.")


# ─── Plot: convergence speed + complexity ─────────────────────────────────────
fig3, (ax_conv, ax_cmplx) = plt.subplots(1, 2, figsize=(16, 6))
fig3.patch.set_facecolor('#080b10')

# ── Left: box plot of convergence speed ──────────────────────────────────────
colors_conv = ['#f87171', '#60a5fa', '#a3e635', '#f59e0b']
names_conv  = list(conv_results.keys())
data_conv   = [conv_results[n] for n in names_conv]

bp = ax_conv.boxplot(data_conv, patch_artist=True,
                     medianprops=dict(color='white', linewidth=2),
                     flierprops=dict(marker='o', color='#6b7888', markersize=4))
for patch, color in zip(bp['boxes'], colors_conv):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'caps']:
    for item in bp[element]:
        item.set_color('#6b7888')

ax_conv.set_xticks(range(1, len(names_conv) + 1))
ax_conv.set_xticklabels(names_conv, rotation=15, ha='right', color='white', fontsize=9)
ax_conv.set_title('Convergence Speed after Channel Switch\n(symbols to re-converge to steady-state MSE)',
                  fontweight='bold', color='white')
ax_conv.set_ylabel('Symbols to converge', color='gray')
ax_conv.set_facecolor('#0e1319')
ax_conv.tick_params(colors='gray')
for spine in ['bottom', 'left']:
    ax_conv.spines[spine].set_color('#2a3441')
ax_conv.spines['top'].set_visible(False)
ax_conv.spines['right'].set_visible(False)
ax_conv.grid(True, alpha=0.12, color='gray', axis='y')

# ── Right: complexity table ───────────────────────────────────────────────────
ax_cmplx.set_facecolor('#080b10')
ax_cmplx.axis('off')

col_labels = ['Optimizer', 'Multiplies', 'Additions', 'Extra Mem', 'Order']
cell_data  = [[r[0], str(r[1]), str(r[2]), str(r[3]), r[4]]
              for r in complexity_rows]

tbl = ax_cmplx.table(
    cellText=cell_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 2.2)

# Header styling
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#1e2d3d')
    tbl[(0, j)].set_text_props(color='#4ade80', fontweight='bold')
    tbl[(0, j)].set_edgecolor('#2a3441')

# Row styling — highlight RLS reference row in amber
row_colors = ['#0e1319', '#151b24']
for i, row in enumerate(cell_data):
    for j in range(len(col_labels)):
        is_rls = 'RLS' in row[0]
        tbl[(i + 1, j)].set_facecolor('#2a1f0e' if is_rls else row_colors[i % 2])
        tbl[(i + 1, j)].set_text_props(color='#f59e0b' if is_rls else 'white')
        tbl[(i + 1, j)].set_edgecolor('#2a3441')

ax_cmplx.set_title(f'Computational Complexity per Symbol Update\n(L={L} taps, B={B}, RLS = reference)',
                   fontweight='bold', color='white', pad=20)

plt.suptitle(f'Convergence & Complexity Analysis  |  L={L} taps  |  SNR=20dB  |  Vehicular A channel',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('convergence_complexity.png', dpi=150, bbox_inches='tight',
            facecolor='#080b10')
plt.show()
print("\nDone. Saved as convergence_complexity.png")
