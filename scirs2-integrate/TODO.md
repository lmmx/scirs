# scirs2-integrate TODO

This module provides numerical integration functionality similar to SciPy's integrate module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Numerical quadrature
  - [x] Basic methods (trapezoid rule, Simpson's rule)
  - [x] Adaptive quadrature for improved accuracy
  - [x] Gaussian quadrature
  - [x] Romberg integration
  - [x] Monte Carlo integration
  - [x] Multiple integration (for higher dimensions)
- [x] Ordinary differential equations (ODE)
  - [x] Euler method
  - [x] Runge-Kutta methods (RK4)
  - [x] Support for first-order ODE systems
- [x] Example code for all integration methods
- [x] Fix implementation issues
  - [x] Fix Gaussian quadrature node/weight calculation (current implementation had scaling issues)
  - [x] Improve Monte Carlo importance sampling stability
  - [x] Handle deep recursion issues better in Romberg integration
- [x] Enhance ODE solvers
  - [x] Variable step-size methods foundations (RK45, RK23)
  - [x] Explicit and implicit methods foundation (BDF implementation)
  - [x] Stiff equation solvers foundation (BDF)
  - [x] Boundary value problems
  - [x] Fix critical implementation issues with variable step and implicit methods:
    - [x] Fix coefficient calculations for RK23 
    - [x] Fix error estimation and step acceptance logic for RK45
    - [x] Revise BDF implementation with more stable numerical method for Newton iterations
    - [x] Add auto-differentiation or better numerical Jacobian calculation for BDF
    - [x] Enable currently ignored tests once fixed
- [x] Add utilities for numerical methods
  - [x] Numerical Jacobian calculation
  - [x] Linear system solver
  - [x] Newton method implementation

## Future Tasks

- [ ] Implement differential algebraic equation (DAE) solvers
  - [ ] Index-1 DAE systems
  - [ ] Higher-index DAE systems with index reduction
  - [ ] Implement BDF methods for DAE systems
- [ ] Add support for partial differential equations (PDE)
  - [ ] Finite difference methods
  - [ ] Finite element methods
  - [ ] Method of lines for time-dependent PDEs
- [ ] Improve error handling and convergence criteria
  - [ ] Better adaptive error control for ODE solvers
  - [ ] Smarter step size selection for stiff problems
- [ ] Add more examples and documentation
  - [ ] Tutorial for common integration problems
  - [ ] Examples for physical system modeling
  - [ ] Comparison with SciPy solutions
- [ ] Additional boundary value problem features
  - [ ] Support for multipoint boundary value problems
  - [ ] More sophisticated mesh adaptation
  - [ ] Support for Robin boundary conditions

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's integrate
  - [ ] Optimize critical numerical routines
  - [ ] Implement SIMD operations for key algorithms
- [ ] Support for parallel and distributed computation
  - [ ] Parallel evaluation of function values in Monte Carlo integration
  - [ ] Parallel solution of independent systems
- [ ] Integration with automatic differentiation for gradient-based methods
  - [ ] True automatic differentiation for Jacobian calculation
  - [ ] Better convergence in implicit methods
- [ ] Support for symbolic manipulation and simplification
  - [ ] Automatic conversion of higher-order ODEs to first-order systems
  - [ ] Symbolic preprocessing of equations
- [ ] Advanced visualization tools for solutions
  - [ ] Phase space plots for ODE systems
  - [ ] Interactive solution exploration
- [ ] Domain-specific solvers for physics, engineering, and finance
  - [ ] Mechanical systems
  - [ ] Chemical kinetics
  - [ ] Circuit simulation
  - [ ] Option pricing and financial modeling