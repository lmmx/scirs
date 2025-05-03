//! Signal processing module
//!
//! This module provides implementations of various signal processing algorithms
//! for filtering, convolution, and spectral analysis.
//!
//! ## Overview
//!
//! * Filtering: FIR and IIR filters, filter design, Savitzky-Golay filter
//! * Convolution and correlation
//! * Spectral analysis and periodograms
//! * Short-time Fourier transform (STFT) and spectrograms
//! * Wavelet transforms
//! * Peak finding and signal measurements
//! * Waveform generation and processing
//! * Resampling and interpolation
//! * Linear Time-Invariant (LTI) systems analysis
//! * Chirp Z-Transform (CZT) for non-uniform frequency sampling
//! * Signal detrending and trend analysis
//! * Hilbert transform and analytic signal analysis
//! * Wavelet transforms (CWT, DWT) and multi-resolution analysis
//!
//! ## Examples
//!
//! ```
//! use scirs2_signal::filter::butter;
//! use scirs2_signal::filter::filtfilt;
//!
//! // Generate a simple signal and apply a Butterworth filter
//! // let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! // let fs = 100.0;  // Sample rate in Hz
//! // let cutoff = 10.0;  // Cutoff frequency in Hz
//! //
//! // let (b, a) = butter(4, cutoff / (fs / 2.0), "lowpass").unwrap();
//! // let filtered = filtfilt(&b, &a, &signal).unwrap();
//! ```

// Export error types
pub mod error;
pub use error::{SignalError, SignalResult};

// Signal processing module structure
pub mod convolve;
pub mod filter;
pub mod peak;
pub mod resample;
pub mod spectral;
pub mod waveforms;
pub mod savgol;
pub mod wavelets;
pub mod lti;
pub mod lti_response;
pub mod czt;
pub mod detrend;
pub mod dwt;
pub mod denoise;
pub mod swt;

// Re-export commonly used functions
pub use convolve::{convolve, correlate, deconvolve};
pub use filter::{bessel, butter, cheby1, cheby2, ellip, filtfilt, firwin, lfilter};
pub use peak::{find_peaks, peak_prominences, peak_widths};
pub use spectral::{periodogram, spectrogram, stft, welch};
pub use waveforms::{chirp, gausspulse, sawtooth, square};

// Savitzky-Golay filtering
pub use savgol::{savgol_coeffs, savgol_filter};

// Wavelet transform functions
pub use wavelets::{cwt, morlet, paul, ricker};
pub use dwt::{Wavelet, WaveletFilters, dwt_decompose, dwt_reconstruct, wavedec, waverec};
pub use swt::{swt, iswt, swt_decompose, swt_reconstruct};

// LTI systems functions
pub use lti::{LtiSystem, TransferFunction, ZerosPoleGain, StateSpace, bode};
pub use lti::system::{tf, zpk, ss, c2d};
pub use lti_response::{impulse_response, step_response, lsim};

// Chirp Z-Transform functions
pub use czt::{czt, czt_points};

// Hilbert transform and related functions
pub mod hilbert;
pub use hilbert::{hilbert, envelope, instantaneous_frequency, instantaneous_phase};

// Detrending functions
pub use detrend::{detrend, detrend_axis, detrend_poly};

// Signal denoising functions
pub use denoise::{denoise_wavelet, ThresholdMethod, ThresholdSelect};

// Signal measurement functions
pub mod measurements;
pub use measurements::{peak_to_peak, peak_to_rms, rms, snr, thd};

// Utility functions for signal processing
pub mod utils;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}