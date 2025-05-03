//! Wavelet transforms
//!
//! This module provides functions for continuous and discrete wavelet transforms,
//! useful for multi-resolution analysis of signals.

use crate::error::{SignalError, SignalResult};
use num_complex::{Complex64, ComplexFloat};
use num_traits::NumCast;
use std::fmt::Debug;

/// Generate a Ricker (Mexican hat) wavelet
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `a` - Width parameter
///
/// # Returns
///
/// * The Ricker wavelet as a vector of length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::ricker;
///
/// // Generate a Ricker wavelet with 100 points and width parameter 4.0
/// let wavelet = ricker(100, 4.0).unwrap();
/// ```
pub fn ricker(points: usize, a: f64) -> SignalResult<Vec<f64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if a <= 0.0 {
        return Err(SignalError::ValueError(
            "width parameter 'a' must be positive".to_string(),
        ));
    }

    // Calculate amplitude factor
    let amplitude = 2.0 / (std::f64::consts::PI.powf(0.25) * (3.0 * a).sqrt());
    let wsq = a * a;

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    for i in 0..points {
        let x = i as f64 - mid_point;
        let xsq = x * x;
        let mod_term = 1.0 - xsq / wsq;
        let gauss = (-xsq / (2.0 * wsq)).exp();
        let value = amplitude * mod_term * gauss;
        wavelet.push(value);
    }

    Ok(wavelet)
}

/// Generate a Morlet wavelet
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `w` - Omega0 parameter (central frequency)
/// * `s` - Scaling factor (bandwidth parameter)
///
/// # Returns
///
/// * The Morlet wavelet as a vector of complex numbers with length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::morlet;
///
/// // Generate a Morlet wavelet with 100 points, central frequency 5.0, and scaling 1.0
/// let wavelet = morlet(100, 5.0, 1.0).unwrap();
/// ```
pub fn morlet(points: usize, w: f64, s: f64) -> SignalResult<Vec<Complex64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }

    if s <= 0.0 {
        return Err(SignalError::ValueError(
            "scaling parameter 's' must be positive".to_string(),
        ));
    }

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    for i in 0..points {
        let t = (i as f64 - mid_point) / s;
        
        // Complex exponential term (oscillation)
        let exp_term = Complex64::new(0.0, w * t).exp();
        
        // Gaussian envelope
        let gauss = (-0.5 * t * t).exp();
        
        // Normalization factor to ensure unit energy
        let norm = (std::f64::consts::PI * s * s).sqrt().recip();
        
        wavelet.push(norm * gauss * exp_term);
    }

    Ok(wavelet)
}

/// Generate a Paul wavelet
///
/// # Arguments
///
/// * `points` - Number of points in the wavelet
/// * `order` - Order of the Paul wavelet (must be positive)
/// * `scale` - Scaling factor
///
/// # Returns
///
/// * The Paul wavelet as a vector of complex numbers with length `points`
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::paul;
///
/// // Generate a Paul wavelet with 100 points, order 4, and scaling 1.0
/// let wavelet = paul(100, 4, 1.0).unwrap();
/// ```
pub fn paul(points: usize, order: usize, scale: f64) -> SignalResult<Vec<Complex64>> {
    if points == 0 {
        return Err(SignalError::ValueError(
            "points must be greater than 0".to_string(),
        ));
    }
    
    if order == 0 {
        return Err(SignalError::ValueError(
            "order must be greater than 0".to_string(),
        ));
    }
    
    if scale <= 0.0 {
        return Err(SignalError::ValueError(
            "scale must be positive".to_string(),
        ));
    }

    // Calculate normalization factor
    let m = order as f64;
    let fact_2m_1 = factorial(2 * order - 1) as f64;
    let fact_m = factorial(order) as f64;
    let norm = (2.0_f64.powf(m) * fact_2m_1 / (std::f64::consts::PI * fact_m)).sqrt();

    // Generate position vector
    let mid_point = (points - 1) as f64 / 2.0;
    let mut wavelet = Vec::with_capacity(points);

    for i in 0..points {
        let t = (i as f64 - mid_point) / scale;
        
        // The Paul wavelet formula is the same for t != 0
        let value = if t != 0.0 {
            let factor = Complex64::new(0.0, 1.0).powf(order as f64);
            let denom = (1.0 - Complex64::new(0.0, t)).powf(order as f64 + 1.0);
            norm * factor / denom
        } else {
            // t == 0 (special case)
            let val = norm * 2.0_f64.powf(m - 1.0) * fact_2m_1 / (fact_m * (2.0 * m - 1.0));
            Complex64::new(val, 0.0)
        };
        
        wavelet.push(value);
    }

    Ok(wavelet)
}

/// Helper function to calculate factorial
fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

/// Continuous wavelet transform
///
/// # Arguments
///
/// * `data` - The input signal (real or complex)
/// * `wavelet` - A function that generates wavelet coefficients. This function should
///   take the number of points and the scale parameter and return a vector of 
///   wavelet coefficients.
/// * `scales` - The scales at which to compute the transform
///
/// # Returns
///
/// * The continuous wavelet transform as a matrix where rows correspond to scales
///   and columns correspond to time points
///
/// # Examples
///
/// ```
/// use scirs2_signal::wavelets::{cwt, ricker};
///
/// // Generate a signal
/// let signal: Vec<f64> = (0..100).map(|i| (i as f64 / 10.0).sin()).collect();
///
/// // Define scales
/// let scales: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0, 16.0];
///
/// // Compute CWT using the Ricker wavelet
/// let result = cwt(&signal, |points, scale| ricker(points, scale), &scales).unwrap();
/// ```
///
/// You can also use it with complex signals:
///
/// ```ignore
/// use scirs2_signal::wavelets::{cwt, morlet};
/// use num_complex::Complex64;
///
/// // Generate a complex signal 
/// let signal: Vec<Complex64> = (0..100)
///     .map(|i| {
///         let t = i as f64 / 10.0;
///         Complex64::new(t.sin(), t.cos())
///     })
///     .collect();
///
/// // Define scales
/// let scales: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0];
///
/// // Compute CWT using the Morlet wavelet with 5.0 as central frequency parameter
/// let result = cwt(&signal, |points, scale| morlet(points, 5.0, scale), &scales).unwrap();
/// ```
pub fn cwt<T, F, W>(
    data: &[T],
    wavelet: F,
    scales: &[f64],
) -> SignalResult<Vec<Vec<Complex64>>>
where
    T: NumCast + Debug + Copy,
    F: Fn(usize, f64) -> SignalResult<Vec<W>>,
    W: Into<Complex64> + Copy,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }
    
    if scales.is_empty() {
        return Err(SignalError::ValueError("Scales array is empty".to_string()));
    }

    // Try to convert to f64 first for real-valued input
    let mut is_complex = false;
    let data_real: Result<Vec<f64>, ()> = data
        .iter()
        .map(|&val| num_traits::cast::cast::<T, f64>(val).ok_or(()))
        .collect();

    // Process data based on type
    let data_complex: Vec<Complex64> = if let Ok(real_data) = data_real {
        // Real data
        real_data.iter().map(|&r| Complex64::new(r, 0.0)).collect()
    } else {
        // Complex data
        is_complex = true;
        data.iter()
            .map(|&val| {
                num_traits::cast::cast::<T, Complex64>(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to Complex64", val))
                })
            })
            .collect::<SignalResult<Vec<_>>>()?
    };

    // Validate scales
    for &scale in scales {
        if scale <= 0.0 {
            return Err(SignalError::ValueError(
                "Scales must be positive".to_string(),
            ));
        }
    }

    // Initialize output
    let mut output = Vec::with_capacity(scales.len());

    // Compute transform for each scale
    for &scale in scales {
        // Determine wavelet size - use at least data.len() points, but limit to reasonable size
        let n = std::cmp::min(data.len() * 10, std::cmp::max(data.len(), 10 * scale as usize));
        
        // Generate wavelet coefficients
        let wavelet_data = wavelet(n, scale)?;
        
        // Convert to complex and take conjugate (for convolution)
        let mut wavelet_complex = Vec::with_capacity(n);
        for &w in &wavelet_data {
            let complex_val: Complex64 = w.into();
            wavelet_complex.push(complex_val.conj());
        }
        
        // Reverse for convolution
        wavelet_complex.reverse();
        
        // Convolve with 'same' mode - choose the right convolution function based on input type
        let convolved = if is_complex {
            convolve_complex_same_complex(&data_complex, &wavelet_complex)
        } else {
            // For real data, we can use the simpler convolution
            convolve_complex_same_real(&data_complex, &wavelet_complex)
        };
        
        output.push(convolved);
    }

    Ok(output)
}

/// Helper function to convolve real signal with complex filter using 'same' mode
/// 
/// Optimized for the CWT case with real input data
fn convolve_complex_same_real(x: &[Complex64], h: &[Complex64]) -> Vec<Complex64> {
    let nx = x.len();
    let nh = h.len();
    let n_out = nx;
    
    // Allocate output buffer
    let mut out = vec![Complex64::new(0.0, 0.0); nx + nh - 1];
    
    // Perform convolution - since h is typically much smaller than x in the CWT case,
    // we optimize by iterating through h in the outer loop for better cache locality
    if nh < nx {
        for j in 0..nh {
            for i in 0..nx {
                out[i + j] += x[i] * h[j];
            }
        }
    } else {
        // Fall back to standard convolution when h is larger
        for i in 0..nx {
            for j in 0..nh {
                out[i + j] += x[i] * h[j];
            }
        }
    }
    
    // Extract the middle part ('same' mode)
    let start = (nh - 1) / 2;
    out.iter().skip(start).take(n_out).copied().collect()
}

/// Helper function to convolve complex signal with complex filter using 'same' mode
/// 
/// Handles fully complex CWT computation
fn convolve_complex_same_complex(x: &[Complex64], h: &[Complex64]) -> Vec<Complex64> {
    let nx = x.len();
    let nh = h.len();
    let n_out = nx;
    
    // Allocate output buffer
    let mut out = vec![Complex64::new(0.0, 0.0); nx + nh - 1];
    
    // Perform convolution - since h is typically much smaller than x in the CWT case,
    // we optimize by iterating through h in the outer loop for better cache locality
    if nh < nx {
        for j in 0..nh {
            for i in 0..nx {
                out[i + j] += x[i] * h[j];
            }
        }
    } else {
        // Fall back to standard convolution when h is larger
        for i in 0..nx {
            for j in 0..nh {
                out[i + j] += x[i] * h[j];
            }
        }
    }
    
    // Extract the middle part ('same' mode)
    let start = (nh - 1) / 2;
    out.iter().skip(start).take(n_out).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ricker_wavelet() {
        // Check that the Ricker wavelet has the right shape
        let points = 100;
        let a = 4.0;
        let wavelet = ricker(points, a).unwrap();
        
        assert_eq!(wavelet.len(), points);
        
        // Check symmetry
        let mid = points / 2;
        for i in 0..mid {
            assert_relative_eq!(wavelet[i], wavelet[points - 1 - i], epsilon = 1e-10);
        }
        
        // Check that the peak is at the middle
        let max_idx = wavelet
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx, mid);
    }

    #[test]
    fn test_morlet_wavelet() {
        // Check that the Morlet wavelet has the right shape
        let points = 100;
        let w = 5.0;
        let s = 1.0;
        let wavelet = morlet(points, w, s).unwrap();
        
        assert_eq!(wavelet.len(), points);
        
        // Check the envelope is symmetric (magnitude should be symmetric)
        let mid = points / 2;
        for i in 0..mid {
            assert_relative_eq!(
                wavelet[i].norm(),
                wavelet[points - 1 - i].norm(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_paul_wavelet() {
        // Check that the Paul wavelet has the right shape
        let points = 100;
        let order = 4;
        let scale = 1.0;
        let wavelet = paul(points, order, scale).unwrap();
        
        assert_eq!(wavelet.len(), points);
        
        // Paul wavelets have complex values with magnitude decreasing from the center
        let mid = points / 2;
        let mut magnitudes = vec![0.0; points];
        for i in 0..points {
            magnitudes[i] = wavelet[i].norm();
        }
        
        // Check peak is near the middle
        let max_idx = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        // Allow some flexibility since Paul wavelets may not peak exactly at center
        assert!((max_idx as isize - mid as isize).abs() <= 5);
    }

    #[test]
    fn test_cwt_with_ricker() {
        // Generate a simple sine wave
        let length = 100;
        let signal: Vec<f64> = (0..length)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
            .collect();
        
        // Define scales
        let scales = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        
        // Compute CWT
        let result = cwt(
            &signal, 
            |points, scale| ricker(points, scale).map(|v| v.into_iter().map(|f| Complex64::new(f, 0.0)).collect()),
            &scales
        ).unwrap();
        
        // Check dimensions
        assert_eq!(result.len(), scales.len());
        for row in &result {
            assert_eq!(row.len(), signal.len());
        }
        
        // Higher scales should capture the sine wave better (at scale close to period)
        let energy: Vec<f64> = result
            .iter()
            .map(|row| row.iter().map(|c| c.norm_sqr()).sum::<f64>())
            .collect();
        
        // Energy should peak at scales close to the signal period (20 samples)
        assert!(energy[3] > energy[0]); // Scale 8 > Scale 1
        assert!(energy[4] > energy[0]); // Scale 16 > Scale 1
    }
}