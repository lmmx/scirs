use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

fn main() {
    println!("LSODA Solver Example - Automatic Stiffness Detection");
    println!("--------------------------------------------------");
    println!("NOTE: The LSODA implementation is still experimental.");
    println!("      It may fail for some problems. This example includes");
    println!("      fallbacks to more stable solvers when LSODA fails.");
    
    // Example: Van der Pol oscillator with varying stiffness
    println!("\nVan der Pol Oscillator with Varying Mu Parameter");
    println!("y'' - μ(1-y²)y' + y = 0");
    println!("As system: y₀' = y₁, y₁' = μ(1-y₀²)y₁ - y₀");
    println!("This problem changes from non-stiff to stiff as μ increases");
    
    // Define an array of mu values to test (increasing stiffness)
    let mu_values = [1.0, 10.0, 100.0, 1000.0];
    
    for &mu in &mu_values {
        println!("\nSolving with μ = {}", mu);
        
        // Define the Van der Pol oscillator with the current mu value
        let van_der_pol = move |_t: f64, y: ArrayView1<f64>| {
            array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]]
        };
        
        // Use LSODA method with appropriate parameters
        // LSODA now directly handles both stiff and non-stiff regions
        let result = solve_ivp(
            van_der_pol,
            [0.0, 20.0],
            array![2.0, 0.0],
            Some(ODEOptions {
                method: ODEMethod::LSODA,
                rtol: 1e-4,
                atol: 1e-6,
                max_steps: 2000,
                // Specify a larger initial step size for better stability
                h0: Some(0.01),
                // Use a much larger min step size to prevent "step size too small" errors
                min_step: Some(1e-4),
                ..Default::default()
            }),
        ).unwrap_or_else(|e| {
            println!("LSODA method failed with error: {}.", e);
            println!("Note: This is expected as LSODA implementation is still experimental.");
            println!("Using a more stable solver instead...");
            
            // Only as a very last resort, try DOP853 method
            println!("Trying DOP853 as a fallback method.");
            solve_ivp(
                van_der_pol,
                [0.0, 20.0],
                array![2.0, 0.0],
                Some(ODEOptions {
                    method: ODEMethod::DOP853,
                    rtol: 1e-4,
                    atol: 1e-6,
                    max_steps: 5000,
                    ..Default::default()
                }),
            ).unwrap()
        });
        
        // Print statistics about the solution
        println!("  Solver method: {:?}", result.method);
        println!("  Steps: {}", result.n_steps);
        println!("  Function evaluations: {}", result.n_eval);
        println!("  Accepted steps: {}", result.n_accepted);
        println!("  Rejected steps: {}", result.n_rejected);
        println!("  Final state: [{:.4}, {:.4}]", 
                 result.y.last().unwrap()[0], 
                 result.y.last().unwrap()[1]);
        println!("  Success: {}", result.success);
        
        // LSODA now returns method switching statistics in the message
        if let Some(msg) = &result.message {
            println!("  {}", msg);
        }
    }
    
    println!("\nLSODA Method Analysis:");
    println!("- LSODA automatically switches between Adams method (for non-stiff regions)");
    println!("  and BDF method (for stiff regions) as needed during integration");
    println!("- For low μ values (1.0), the system is mostly non-stiff and should use Adams method");
    println!("- For high μ values (1000.0), the system is very stiff and should use BDF method");
    println!("- For intermediate values, we should see method switching during integration");
}