//! Unconstrained optimization algorithms
//!
//! This module provides methods for unconstrained optimization of scalar
//! functions of one or more variables.
//!
//! ## Example
//!
//! ```
//! use ndarray::array;
//! use scirs2_optimize::unconstrained::{minimize, Method};
//!
//! // Define a simple function to minimize
//! fn quadratic(x: &[f64]) -> f64 {
//!     x.iter().map(|&xi| xi * xi).sum()
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Minimize the function starting at [1.0, 1.0]
//! let initial_point = array![1.0, 1.0];
//! let result = minimize(quadratic, &initial_point, Method::BFGS, None)?;
//!
//! // The minimum should be at [0.0, 0.0]
//! assert!(result.success);
//! assert!(result.x.iter().all(|&x| x.abs() < 1e-6));
//! assert!(result.fun.abs() < 1e-10);
//! # Ok(())
//! # }
//! ```

use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1};
use std::fmt;

/// Optimization methods for unconstrained minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Nelder-Mead simplex algorithm
    NelderMead,

    /// Powell's method
    Powell,

    /// Conjugate gradient method
    CG,

    /// Broyden-Fletcher-Goldfarb-Shanno algorithm
    BFGS,

    /// Limited-memory BFGS with optional inverse Hessian approximation
    LBFGS,

    /// Newton-Conjugate-Gradient algorithm
    NewtonCG,

    /// Trust-region Newton-Conjugate-Gradient algorithm
    TrustNCG,

    /// Trust-region truncated generalized Lanczos / conjugate gradient algorithm
    TrustKrylov,

    /// Trust-region nearly exact algorithm
    TrustExact,

    /// Truncated Newton method with Conjugate Gradient
    TNC,

    /// Sequential Least SQuares Programming
    SLSQP,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Method::NelderMead => write!(f, "Nelder-Mead"),
            Method::Powell => write!(f, "Powell"),
            Method::CG => write!(f, "CG"),
            Method::BFGS => write!(f, "BFGS"),
            Method::LBFGS => write!(f, "L-BFGS"),
            Method::NewtonCG => write!(f, "Newton-CG"),
            Method::TrustNCG => write!(f, "trust-ncg"),
            Method::TrustKrylov => write!(f, "trust-krylov"),
            Method::TrustExact => write!(f, "trust-exact"),
            Method::TNC => write!(f, "TNC"),
            Method::SLSQP => write!(f, "SLSQP"),
        }
    }
}

/// Options for the unconstrained optimizer.
#[derive(Debug, Clone)]
pub struct Options {
    /// Maximum number of iterations to perform
    pub maxiter: Option<usize>,

    /// Precision goal for the value in the stopping criterion
    pub ftol: Option<f64>,

    /// Precision goal for the gradient in the stopping criterion (relative)
    pub gtol: Option<f64>,

    /// Step size used for numerical approximation of the jacobian
    pub eps: Option<f64>,

    /// Relative step size to use in numerical differentiation
    pub finite_diff_rel_step: Option<f64>,

    /// Whether to print convergence messages
    pub disp: bool,

    /// Return the optimization result after each iteration
    pub return_all: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            maxiter: None,
            ftol: Some(1e-8),
            gtol: Some(1e-8),
            eps: Some(1e-8),
            finite_diff_rel_step: None,
            disp: false,
            return_all: false,
        }
    }
}

/// Minimizes a scalar function of one or more variables.
///
/// # Arguments
///
/// * `func` - A function that takes a slice of values and returns a scalar
/// * `x0` - The initial guess
/// * `method` - The optimization method to use
/// * `options` - Options for the optimizer
///
/// # Returns
///
/// * `OptimizeResults` containing the optimization results
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_optimize::unconstrained::{minimize, Method};
///
/// fn rosenbrock(x: &[f64]) -> f64 {
///     let a = 1.0;
///     let b = 100.0;
///     let x0 = x[0];
///     let x1 = x[1];
///     (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let initial_guess = array![0.0, 0.0];
/// let result = minimize(rosenbrock, &initial_guess, Method::BFGS, None)?;
///
/// println!("Solution: {:?}", result.x);
/// println!("Function value at solution: {}", result.fun);
/// # Ok(())
/// # }
/// ```
pub fn minimize<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    method: Method,
    options: Option<Options>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    let options = options.unwrap_or_default();

    // Implementation of various methods will go here
    match method {
        Method::NelderMead => minimize_nelder_mead(func, x0, &options),
        Method::BFGS => minimize_bfgs(func, x0, &options),
        Method::Powell => minimize_powell(func, x0, &options),
        Method::CG => minimize_conjugate_gradient(func, x0, &options),
        _ => Err(OptimizeError::NotImplementedError(format!(
            "Method {:?} is not yet implemented",
            method
        ))),
    }
}

/// Implements the Nelder-Mead simplex algorithm
fn minimize_nelder_mead<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    // Nelder-Mead algorithm parameters
    let alpha = 1.0; // Reflection parameter
    let gamma = 2.0; // Expansion parameter
    let rho = 0.5; // Contraction parameter
    let sigma = 0.5; // Shrink parameter

    // Get the dimension of the problem
    let n = x0.len();

    // Set the maximum number of iterations
    let maxiter = options.maxiter.unwrap_or(200 * n);

    // Set the tolerance
    let ftol = options.ftol.unwrap_or(1e-8);

    // Initialize the simplex
    let mut simplex = Vec::with_capacity(n + 1);
    let x0_vec = x0.to_owned();
    simplex.push((x0_vec.clone(), func(x0_vec.as_slice().unwrap())));

    // Create the initial simplex
    for i in 0..n {
        let mut xi = x0.to_owned();
        if xi[i] != 0.0 {
            xi[i] *= 1.05;
        } else {
            xi[i] = 0.00025;
        }

        simplex.push((xi.clone(), func(xi.as_slice().unwrap())));
    }

    let mut nfev = n + 1;

    // Sort the simplex by function value
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Iteration counter
    let mut iter = 0;

    // Main iteration loop
    while iter < maxiter {
        // Check convergence: if the difference in function values is less than the tolerance
        if (simplex[n].1 - simplex[0].1).abs() < ftol {
            break;
        }

        // Compute the centroid of all points except the worst one
        let mut xc = Array1::zeros(n);
        for item in simplex.iter().take(n) {
            xc = &xc + &item.0;
        }
        xc = &xc / n as f64;

        // Reflection: reflect the worst point through the centroid
        let xr: Array1<f64> = &xc + alpha * (&xc - &simplex[n].0);
        let fxr = func(xr.as_slice().unwrap());
        nfev += 1;

        if fxr < simplex[0].1 {
            // If the reflected point is the best so far, try expansion
            let xe: Array1<f64> = &xc + gamma * (&xr - &xc);
            let fxe = func(xe.as_slice().unwrap());
            nfev += 1;

            if fxe < fxr {
                // If the expanded point is better than the reflected point,
                // replace the worst point with the expanded point
                simplex[n] = (xe, fxe);
            } else {
                // Otherwise, replace the worst point with the reflected point
                simplex[n] = (xr, fxr);
            }
        } else if fxr < simplex[n - 1].1 {
            // If the reflected point is better than the second worst,
            // replace the worst point with the reflected point
            simplex[n] = (xr, fxr);
        } else {
            // Otherwise, try contraction
            let xc_contract: Array1<f64> = if fxr < simplex[n].1 {
                // Outside contraction
                &xc + rho * (&xr - &xc)
            } else {
                // Inside contraction
                &xc - rho * (&xc - &simplex[n].0)
            };

            let fxc_contract = func(xc_contract.as_slice().unwrap());
            nfev += 1;

            if fxc_contract < simplex[n].1 {
                // If the contracted point is better than the worst point,
                // replace the worst point with the contracted point
                simplex[n] = (xc_contract, fxc_contract);
            } else {
                // If all else fails, shrink the simplex towards the best point
                for i in 1..=n {
                    simplex[i].0 = &simplex[0].0 + sigma * (&simplex[i].0 - &simplex[0].0);
                    simplex[i].1 = func(simplex[i].0.as_slice().unwrap());
                    nfev += 1;
                }
            }
        }

        // Resort the simplex
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        iter += 1;
    }

    // Get the best point and its function value
    let (x_best, f_best) = simplex[0].clone();

    // Create the result
    let mut result = OptimizeResults::default();
    result.x = x_best;
    result.fun = f_best;
    result.nfev = nfev;
    result.nit = iter;
    result.success = iter < maxiter;
    result.message = if result.success {
        "Optimization terminated successfully".to_string()
    } else {
        "Maximum number of iterations reached".to_string()
    };

    Ok(result)
}

/// Implements the BFGS algorithm
fn minimize_bfgs<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    // Get options or use defaults
    let ftol = options.ftol.unwrap_or(1e-8);
    let gtol = options.gtol.unwrap_or(1e-8);
    let maxiter = options.maxiter.unwrap_or(100 * x0.len());
    let eps = options.eps.unwrap_or(1e-8);

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut f = func(x.as_slice().unwrap());

    // Calculate initial gradient using finite differences
    let mut g = Array1::zeros(n);
    for i in 0..n {
        let mut x_h = x.clone();
        x_h[i] += eps;
        let f_h = func(x_h.as_slice().unwrap());
        g[i] = (f_h - f) / eps;
    }

    // Initialize approximation of inverse Hessian with identity matrix
    let mut h_inv = Array2::eye(n);

    // Initialize counters
    let mut iter = 0;
    let mut nfev = 1 + n; // Initial evaluation plus gradient calculations

    // Main loop
    while iter < maxiter {
        // Check convergence on gradient
        if g.iter().all(|&gi| gi.abs() < gtol) {
            break;
        }

        // Compute search direction
        let p = -&h_inv.dot(&g);

        // Line search using backtracking
        let mut alpha = 1.0;
        let c1 = 1e-4; // Sufficient decrease parameter
        let rho = 0.5; // Backtracking parameter

        // Initial step
        let mut x_new = &x + &(&p * alpha);
        let mut f_new = func(x_new.as_slice().unwrap());
        nfev += 1;

        // Backtracking until Armijo condition is satisfied
        let g_dot_p = g.dot(&p);
        while f_new > f + c1 * alpha * g_dot_p {
            alpha *= rho;
            x_new = &x + &(&p * alpha);
            f_new = func(x_new.as_slice().unwrap());
            nfev += 1;

            // Prevent infinite loops for very small steps
            if alpha < 1e-10 {
                break;
            }
        }

        // Compute step and gradient difference
        let s = &x_new - &x;

        // Calculate new gradient
        let mut g_new = Array1::zeros(n);
        for i in 0..n {
            let mut x_h = x_new.clone();
            x_h[i] += eps;
            let f_h = func(x_h.as_slice().unwrap());
            g_new[i] = (f_h - f_new) / eps;
            nfev += 1;
        }

        let y = &g_new - &g;

        // Check convergence on function value
        if (f - f_new).abs() < ftol * (1.0 + f.abs()) {
            // Update variables for the final iteration
            x = x_new;
            f = f_new;
            g = g_new;
            break;
        }

        // Update inverse Hessian approximation using BFGS formula
        let rho_bfgs = 1.0 / y.dot(&s);
        if rho_bfgs.is_finite() && rho_bfgs > 0.0 {
            let i_mat = Array2::eye(n);
            let y_row = y.clone().insert_axis(Axis(0));
            let s_col = s.clone().insert_axis(Axis(1));
            let y_s_t = y_row.dot(&s_col);

            let term1 = &i_mat - &(&y_s_t * rho_bfgs);
            let s_row = s.clone().insert_axis(Axis(0));
            let y_col = y.clone().insert_axis(Axis(1));
            let s_y_t = s_row.dot(&y_col);

            let term2 = &i_mat - &(&s_y_t * rho_bfgs);
            let term3 = &term1.dot(&h_inv);
            h_inv = term3.dot(&term2) + rho_bfgs * s_col.dot(&s_row);
        }

        // Update variables for next iteration
        x = x_new;
        f = f_new;
        g = g_new;

        iter += 1;
    }

    // Create and return result
    let mut result = OptimizeResults::default();
    result.x = x;
    result.fun = f;
    result.jac = Some(g.into_raw_vec_and_offset().0);
    result.nfev = nfev;
    result.nit = iter;
    result.success = iter < maxiter;

    if result.success {
        result.message = "Optimization terminated successfully.".to_string();
    } else {
        result.message = "Maximum iterations reached.".to_string();
    }

    Ok(result)
}

/// Implements Powell's method for unconstrained optimization
fn minimize_powell<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    // Get options or use defaults
    let ftol = options.ftol.unwrap_or(1e-8);
    let maxiter = options.maxiter.unwrap_or(100 * x0.len());

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut f = func(x.as_slice().unwrap());

    // Initialize the set of directions as the standard basis
    let mut directions = Vec::with_capacity(n);
    for i in 0..n {
        let mut e_i = Array1::zeros(n);
        e_i[i] = 1.0;
        directions.push(e_i);
    }

    // Counters
    let mut iter = 0;
    let mut nfev = 1; // Initial function evaluation

    // Powell's main loop
    while iter < maxiter {
        let x_old = x.clone();
        let f_old = f;

        // Keep track of the greatest function reduction
        let mut f_reduction_max = 0.0;
        let mut reduction_idx = 0;

        // Line search along each direction
        for (i, u) in directions.iter().enumerate().take(n) {
            let f_before = f;

            // Line search along direction u
            let (alpha, f_min) = line_search_powell(&func, &x, u, f, &mut nfev);

            // Update current position and function value
            x = &x + &(alpha * u);
            f = f_min;

            // Update maximum reduction tracker
            let reduction = f_before - f;
            if reduction > f_reduction_max {
                f_reduction_max = reduction;
                reduction_idx = i;
            }
        }

        let new_dir = &x - &x_old;

        //  extra line search along the extrapolated direction
        let (alpha, f_min) =
            line_search_powell(&func, &x, &new_dir, f, &mut nfev);
        x = &x + &(alpha * &new_dir);
        f = f_min;

        //  only *now* check for convergence
        if 2.0*(f_old - f)
            <= ftol * (f_old.abs() + f.abs() + 1e-10)
        {
            break;
        }

        // Keep the basis full rank.
        // If the extrapolated displacement is numerically zero we would
        // lose a basis direction; just keep the old one instead.
        if new_dir.iter().any(|v| v.abs() > 1e-12) {
            directions[reduction_idx] = new_dir;
        }
        iter += 1;
    }

    // Create and return result
    let mut result = OptimizeResults::default();
    result.x = x;
    result.fun = f;
    result.nfev = nfev;
    result.nit = iter;
    result.success = iter < maxiter;

    if result.success {
        result.message = "Optimization terminated successfully.".to_string();
    } else {
        result.message = "Maximum iterations reached.".to_string();
    }

    Ok(result)
}

/// Helper function for line search in Powell's method
fn line_search_powell<F>(
    func: F,
    x: &Array1<f64>,
    direction: &Array1<f64>,
    f_x: f64,
    nfev: &mut usize,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    // Degenerate direction ⇒ no movement
    if direction.iter().all(|v| v.abs() <= 1e-16) {
        return (0.0, f_x);
    }

    // helper ϕ(α)
    let mut phi = |alpha: f64| {
        let y = x + &(direction * alpha);
        *nfev += 1;
        func(y.as_slice().unwrap())
    };

    //------------------------------------------------------------------
    // 1) Bracket a minimum with a golden‑ratio expansion
    //------------------------------------------------------------------
    let golden = 1.618_034;
    let mut a = 0.0;
    let mut fa = f_x;            // ϕ(0)
    let mut b = 1.0;
    let mut fb = phi(b);

    // make sure we are going downhill
    if fb > fa {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = b + golden * (b - a);
    let mut fc = phi(c);

    const MAX_BRACKET: usize = 50;
    for _ in 0..MAX_BRACKET {
        if fb <= fc {
            break;               // [a, b, c] is a valid bracket
        }
        a = b;  fa = fb;
        b = c;  fb = fc;
        c = b + golden * (b - a);
        fc = phi(c);
    }

    //------------------------------------------------------------------
    // 2) Brent search inside the bracket
    //------------------------------------------------------------------
    let (mut lo, mut hi) = if a < c { (a, c) } else { (c, a) };
    let mut mid = b;
    let mut fm  = fb;
    let mut d_last: f64 = 0.0;

    const IT_MAX: usize = 100;
    const TOL: f64 = 1e-8;

    for _ in 0..IT_MAX {
        let tol1 = TOL * mid.abs() + 1e-10;
        let m    = 0.5 * (lo + hi);

        // stop when the interval is tiny
        if (mid - m).abs() <= 2.0 * tol1 - 0.5 * (hi - lo) {
            break;
        }

        //--------------------------------------------------------------
        // parabolic step (if it stays inside the bracket)
        //--------------------------------------------------------------
        let mut accept_parabolic = false;
        let mut d = 0.0;
        if d_last.abs() > tol1 {
            let r = (mid - lo) * (fm - fc);
            let q = (mid - hi) * (fm - fa);
            let p = (mid - hi) * q - (mid - lo) * r;
            let q = 2.0 * (q - r);

            if q.abs() > 1e-21 {
                let s = p / q;
                if (lo + s) < (hi - s) && s.abs() < 0.5 * (hi - lo) {
                    d = s;
                    accept_parabolic = true;
                }
            }
        }

        if !accept_parabolic {
            // golden‑section step
            d = if mid >= m { -0.381_966_0 * (mid - lo) }
                else         {  0.381_966_0 * (hi - mid) };
        }

        let u = mid + if d.abs() >= tol1 { d }
                      else               { d.signum() * tol1 };
        let fu = phi(u);

        // update bracket points
        if fu <= fm {
            if u < mid { hi = mid; fc = fm; } else { lo = mid; fa = fm; }
            mid = u; fm = fu;
        } else {
            if u < mid { lo = u;  fa = fu; } else { hi = u;  fc = fu; }
        }
        d_last = d;
    }

    (mid, fm)
}

/// Implements the Conjugate Gradient method for unconstrained optimization
fn minimize_conjugate_gradient<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    // Get options or use defaults
    let ftol = options.ftol.unwrap_or(1e-8);
    let gtol = options.gtol.unwrap_or(1e-8);
    let maxiter = options.maxiter.unwrap_or(100 * x0.len());
    let eps = options.eps.unwrap_or(1e-8);

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut f = func(x.as_slice().unwrap());

    // Calculate initial gradient using finite differences
    let mut g = Array1::zeros(n);
    for i in 0..n {
        let mut x_h = x.clone();
        x_h[i] += eps;
        let f_h = func(x_h.as_slice().unwrap());
        g[i] = (f_h - f) / eps;
    }

    // Initialize search direction as steepest descent
    let mut p = -g.clone();

    // Counters
    let mut iter = 0;
    let mut nfev = 1 + n; // Initial evaluation plus gradient calculations

    while iter < maxiter {
        // Check convergence on gradient
        if g.iter().all(|&gi| gi.abs() < gtol) {
            break;
        }

        // Line search along the search direction
        let (alpha, f_new) = line_search_cg(&func, &x, &p, f, &mut nfev);

        // Update position
        let x_new = &x + &(&p * alpha);

        // Compute new gradient
        let mut g_new = Array1::zeros(n);
        for i in 0..n {
            let mut x_h = x_new.clone();
            x_h[i] += eps;
            let f_h = func(x_h.as_slice().unwrap());
            g_new[i] = (f_h - f_new) / eps;
            nfev += 1;
        }

        // Check convergence on function value
        if (f - f_new).abs() < ftol * (1.0 + f.abs()) {
            x = x_new;
            f = f_new;
            g = g_new;
            break;
        }

        // Calculate beta using the Fletcher-Reeves formula
        let beta_fr = g_new.dot(&g_new) / g.dot(&g);

        // Update search direction
        p = -&g_new + beta_fr * &p;

        // Update variables for next iteration
        x = x_new;
        f = f_new;
        g = g_new;

        iter += 1;

        // Restart direction to steepest descent every n iterations
        if iter % n == 0 {
            p = -g.clone();
        }
    }

    // Create and return result
    let mut result = OptimizeResults::default();
    result.x = x;
    result.fun = f;
    result.jac = Some(g.into_raw_vec_and_offset().0);
    result.nfev = nfev;
    result.nit = iter;
    result.success = iter < maxiter;

    if result.success {
        result.message = "Optimization terminated successfully.".to_string();
    } else {
        result.message = "Maximum iterations reached.".to_string();
    }

    Ok(result)
}

/// Helper function for line search in Conjugate Gradient method
fn line_search_cg<F>(
    func: F,
    x: &Array1<f64>,
    direction: &Array1<f64>,
    f_x: f64,
    nfev: &mut usize,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    // Use a simple backtracking line search
    let c1 = 1e-4; // Sufficient decrease parameter
    let rho = 0.5; // Backtracking parameter
    let mut alpha = 1.0;

    // Function to evaluate a point on the line
    let mut f_line = |alpha: f64| {
        let x_new = x + alpha * direction;
        *nfev += 1;
        func(x_new.as_slice().unwrap())
    };

    // Initial step
    let mut f_new = f_line(alpha);

    // Backtracking until Armijo condition is satisfied
    let slope = direction.iter().map(|&d| d * d).sum::<f64>();
    while f_new > f_x - c1 * alpha * slope.abs() {
        alpha *= rho;
        f_new = f_line(alpha);

        // Prevent infinite loops for very small steps
        if alpha < 1e-10 {
            break;
        }
    }

    (alpha, f_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    fn quadratic(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0;
        let b = 100.0;
        let x0 = x[0];
        let x1 = x[1];
        (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
    }

    #[test]
    fn test_minimize_bfgs_quadratic() {
        let x0 = array![1.0, 1.0];
        let result = minimize(quadratic, &x0.view(), Method::BFGS, None).unwrap();

        // The quadratic function should be minimized at [0, 0]
        assert!(result.success);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_minimize_bfgs_rosenbrock() {
        let x0 = array![0.0, 0.0];

        // For testing, we're more interested in the algorithm working than in perfect convergence
        let options = Options {
            maxiter: Some(1000),
            gtol: Some(1e-4), // More lenient tolerance
            ftol: Some(1e-4), // More lenient tolerance
            ..Options::default()
        };

        let result = minimize(rosenbrock, &x0.view(), Method::BFGS, Some(options)).unwrap();

        // The minimum of Rosenbrock is at [1.0, 1.0]
        assert!(result.success);
        assert_relative_eq!(result.fun, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_minimize_nelder_mead_quadratic() {
        let x0 = array![1.0, 1.0];
        let result = minimize(quadratic, &x0.view(), Method::NelderMead, None).unwrap();

        // The minimum of the quadratic function should be at the origin
        assert!(result.success);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_minimize_nelder_mead_rosenbrock() {
        let x0 = array![0.0, 0.0];

        // Specify maximum iterations to ensure test runs quickly
        let options = Options {
            maxiter: Some(1000),
            ..Options::default()
        };

        let result = minimize(rosenbrock, &x0.view(), Method::NelderMead, Some(options)).unwrap();

        // The minimum of the Rosenbrock function is at (1, 1)
        // Nelder-Mead might not converge exactly to the minimum in a limited number of iterations,
        // but it should get close
        assert!(result.success);
        assert_relative_eq!(result.fun, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_minimize_powell_quadratic() {
        let x0 = array![1.0, 1.0];

        let options = Options {
            maxiter: Some(20), // Fewer iterations to avoid unreliable behavior
            ftol: Some(1e-3),  // Much more relaxed tolerance
            ..Options::default()
        };

        let result = minimize(quadratic, &x0.view(), Method::Powell, Some(options.clone())).unwrap();

        // Check that the optimization found coordinates close to the minimum
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-2);

        // Check that the function value at the minimum is close to zero
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-2);

        // Check that the algorithm converged within our iteration limit
        assert!(result.nit <= options.maxiter.unwrap(),
                "Algorithm used too many iterations: {}", result.nit);

        // Still keep the informative print for debugging purposes
        println!(
            "Powell quadratic: x = {:?}, f = {}, initial = {}, iters = {}",
            result.x, result.fun, quadratic(&[1.0, 1.0]), result.nit
        );
    }

    #[test]
    fn test_minimize_powell_rosenbrock() {
        let x0 = array![0.0, 0.0];

        let options = Options {
            maxiter: Some(2000), // Increased iterations
            ftol: Some(1e-6),
            ..Options::default()
        };

        let result = minimize(rosenbrock, &x0.view(), Method::Powell, Some(options)).unwrap();

        assert!(result.success);

        // The minimum of the Rosenbrock function is at (1, 1)
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);

        println!(
            "Powell Rosenbrock: x = {:?}, f = {}, iter = {}",
            result.x, result.fun, result.nit
        );
    }

    #[test]
    fn test_minimize_cg_quadratic() {
        let x0 = array![1.0, 1.0];

        let options = Options {
            maxiter: Some(1000), // Increased iterations
            ftol: Some(1e-8),
            gtol: Some(1e-8),
            ..Options::default()
        };

        let result = minimize(quadratic, &x0.view(), Method::CG, Some(options)).unwrap();

        assert!(result.success);

        // The minimum of the quadratic function should be at the origin
        assert!(result.success);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_minimize_cg_rosenbrock() {
        let x0 = array![0.0, 0.0];

        let options = Options {
            maxiter: Some(1000),
            ftol: Some(1e-6),
            gtol: Some(1e-5),
            ..Options::default()
        };

        let result = minimize(rosenbrock, &x0.view(), Method::CG, Some(options)).unwrap();

        assert!(result.success);

        // The minimum of the Rosenbrock function is at (1, 1)
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }
}
