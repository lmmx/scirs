# scirs2-optimize TODO

This module provides optimization algorithms similar to SciPy's optimize module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Unconstrained minimization (Nelder-Mead, BFGS, Powell, Conjugate Gradient)
- [x] Constrained minimization (SLSQP, Trust-region constrained)
- [x] Least squares minimization (Levenberg-Marquardt, Trust Region Reflective)
- [x] Root finding (Powell, Broyden's methods, Anderson, Krylov)
- [x] Integration with existing optimization libraries (argmin)
- [x] Bounds support for all unconstrained minimization methods:
  - Powell's method with boundary-respecting line search
  - Nelder-Mead with boundary projection for simplex operations
  - BFGS with projected gradients and modified gradient calculations at boundaries
  - Conjugate Gradient with projected search directions

## Future Tasks

- [x] Fix any warnings in the current implementation
- [x] Support for bounds in unconstrained optimization algorithms
- [x] Add L-BFGS-B algorithm for bound-constrained optimization
- [x] Add L-BFGS algorithm for large-scale optimization
- [x] Add TrustNCG (Trust-region Newton-Conjugate-Gradient) algorithm
- [ ] Add more algorithm options and variants
- [ ] Improve convergence criteria and control
- [ ] Add more examples and test cases
- [ ] Enhance documentation with theoretical background
- [ ] Performance optimizations for high-dimensional problems
- [ ] Implement global optimization methods
- [ ] Add visualization tools for optimization process
- [ ] Improve error handling and diagnostics

## Long-term Goals

- [ ] Create a unified API for all optimization methods
- [ ] Support for parallel and distributed optimization
- [ ] Integration with automatic differentiation for gradient-based methods
- [ ] Support for stochastic optimization methods
- [ ] Implement specialized optimizers for machine learning