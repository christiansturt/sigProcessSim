"""DSP framework for dual-path complex signal processing using scipy.signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class DSPConfig:
    """Configuration for the dual-path DSP framework."""

    core_clock_hz: float = 150e6
    duration_s: float = 5e-6
    carrier_hz: float = 1.2e9

    @property
    def sample_rate_hz(self) -> float:
        return 64.0 * self.core_clock_hz


class DualPathDSP:
    """Two-path DSP chain: BFM0 and BFM1."""

    def __init__(self, config: Optional[DSPConfig] = None):
        self.config = config or DSPConfig()

    def generate_input_signals(
        self,
        amp0: float = 1.0,
        amp1: float = 1.0,
        phase0_rad: float = 0.0,
        phase1_rad: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 5us (configurable) complex I/Q carriers for BFM0 and BFM1."""
        fs = self.config.sample_rate_hz
        n_samples = int(round(self.config.duration_s * fs))
        t = np.arange(n_samples) / fs

        bfm0 = amp0 * np.exp(1j * (2.0 * np.pi * self.config.carrier_hz * t + phase0_rad))
        bfm1 = amp1 * np.exp(1j * (2.0 * np.pi * self.config.carrier_hz * t + phase1_rad))
        return t, bfm0, bfm1

    @staticmethod
    def apply_time_delay(x: np.ndarray, fs: float, delta_s: float) -> np.ndarray:
        """Stage 1: time delay with zero-filled boundaries (supports fractional delay)."""
        t = np.arange(x.size) / fs
        t_delayed = t - delta_s

        y_real = np.interp(t_delayed, t, np.real(x), left=0.0, right=0.0)
        y_imag = np.interp(t_delayed, t, np.imag(x), left=0.0, right=0.0)
        return y_real + 1j * y_imag

    @staticmethod
    def complex_multiply(x: np.ndarray, i_mult: float, q_mult: float) -> np.ndarray:
        """Stage 2: complex multiply by configurable I/Q value (i_mult + j*q_mult)."""
        return x * (i_mult + 1j * q_mult)

    @staticmethod
    def decimate_by_2(x: np.ndarray) -> np.ndarray:
        """Stage 3: decimate by 2 with anti-aliasing filter."""
        return signal.decimate(x, 2, ftype="fir", zero_phase=True)

    @staticmethod
    def phase_delta(sig0: np.ndarray, sig1: np.ndarray) -> np.ndarray:
        """Stage 4: per-sample phase delta between paths in radians (wrapped to [-pi, pi])."""
        return np.angle(sig1) - np.angle(sig0)

    @staticmethod
    def add_signals(sig0: np.ndarray, sig1: np.ndarray) -> np.ndarray:
        """Stage 5: add two complex signals."""
        return sig0 + sig1

    def run_stages_1_to_4(
        self,
        bfm0: np.ndarray,
        bfm1: np.ndarray,
        delta0_s: float,
        delta1_s: float,
        i_mult: float,
        q_mult: float,
    ) -> Dict[str, np.ndarray]:
        """Run stages 1-4 in parallel on BFM0/BFM1 and return intermediate results."""
        fs = self.config.sample_rate_hz

        s1_0 = self.apply_time_delay(bfm0, fs, delta0_s)
        s1_1 = self.apply_time_delay(bfm1, fs, delta1_s)

        s2_0 = self.complex_multiply(s1_0, i_mult, q_mult)
        s2_1 = self.complex_multiply(s1_1, i_mult, q_mult)

        s3_0 = self.decimate_by_2(s2_0)
        s3_1 = self.decimate_by_2(s2_1)

        phase_delta = self.phase_delta(s3_0, s3_1)

        fs_dec = fs / 2.0
        t_dec = np.arange(s3_0.size) / fs_dec

        return {
            "t_dec": t_dec,
            "fs_dec": np.array([fs_dec]),
            "bfm0": s3_0,
            "bfm1": s3_1,
            "phase_delta": phase_delta,
        }

    def run_full_chain(
        self,
        delta0_s: float = 0.0,
        delta1_s: float = 0.0,
        i_mult: float = 1.0,
        q_mult: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Run stages 1-5 and return outputs."""
        _, bfm0, bfm1 = self.generate_input_signals()
        out_1_4 = self.run_stages_1_to_4(bfm0, bfm1, delta0_s, delta1_s, i_mult, q_mult)
        combined = self.add_signals(out_1_4["bfm0"], out_1_4["bfm1"])

        out_1_4["combined"] = combined
        return out_1_4

    @staticmethod
    def _fft_mag_db(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        n = x.size
        spec = np.fft.fftshift(np.fft.fft(x))
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
        mag_db = 20.0 * np.log10(np.maximum(np.abs(spec), 1e-12))
        return freqs, mag_db

    def plot_two_paths(
        self,
        t: np.ndarray,
        sig0: np.ndarray,
        sig1: np.ndarray,
        fs: float,
        title: str = "BFM0/BFM1 Signals",
    ) -> Tuple[Any, np.ndarray]:
        """Plot time-domain I/Q (left) and DFT magnitude (right) for BFM0 and BFM1."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        fig.suptitle(title)

        axes[0, 0].plot(t * 1e6, np.real(sig0), label="BFM0 I")
        axes[0, 0].plot(t * 1e6, np.imag(sig0), label="BFM0 Q", alpha=0.8)
        axes[0, 0].set_title("BFM0 Time Domain")
        axes[0, 0].set_xlabel("Time (us)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        axes[1, 0].plot(t * 1e6, np.real(sig1), label="BFM1 I")
        axes[1, 0].plot(t * 1e6, np.imag(sig1), label="BFM1 Q", alpha=0.8)
        axes[1, 0].set_title("BFM1 Time Domain")
        axes[1, 0].set_xlabel("Time (us)")
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        f0, m0 = self._fft_mag_db(sig0, fs)
        f1, m1 = self._fft_mag_db(sig1, fs)

        axes[0, 1].plot(f0 / 1e6, m0)
        axes[0, 1].set_title("BFM0 DFT Magnitude")
        axes[0, 1].set_xlabel("Frequency (MHz)")
        axes[0, 1].set_ylabel("Magnitude (dB)")
        axes[0, 1].grid(True)

        axes[1, 1].plot(f1 / 1e6, m1)
        axes[1, 1].set_title("BFM1 DFT Magnitude")
        axes[1, 1].set_xlabel("Frequency (MHz)")
        axes[1, 1].set_ylabel("Magnitude (dB)")
        axes[1, 1].grid(True)

        return fig, axes

    def plot_combined_signal(
        self,
        t: np.ndarray,
        combined: np.ndarray,
        fs: float,
        title: str = "Stage 5 Combined Signal",
    ) -> Tuple[Any, np.ndarray]:
        """Plot combined signal time-domain I/Q and DFT magnitude."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
        fig.suptitle(title)

        axes[0].plot(t * 1e6, np.real(combined), label="Combined I")
        axes[0].plot(t * 1e6, np.imag(combined), label="Combined Q", alpha=0.8)
        axes[0].set_title("Combined Time Domain")
        axes[0].set_xlabel("Time (us)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        axes[0].legend()

        f, m = self._fft_mag_db(combined, fs)
        axes[1].plot(f / 1e6, m)
        axes[1].set_title("Combined DFT Magnitude")
        axes[1].set_xlabel("Frequency (MHz)")
        axes[1].set_ylabel("Magnitude (dB)")
        axes[1].grid(True)

        return fig, axes


if __name__ == "__main__":
    dsp = DualPathDSP()

    # Example run
    outputs = dsp.run_full_chain(
        delta0_s=10e-9,
        delta1_s=30e-9,
        i_mult=0.9,
        q_mult=0.2,
    )

    t_dec = outputs["t_dec"]
    fs_dec = float(outputs["fs_dec"][0])

    try:
        import matplotlib.pyplot as plt

        dsp.plot_two_paths(t_dec, outputs["bfm0"], outputs["bfm1"], fs_dec, title="After Stages 1-4")
        dsp.plot_combined_signal(t_dec, outputs["combined"], fs_dec)
        plt.show()
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping plots in the demo.")
