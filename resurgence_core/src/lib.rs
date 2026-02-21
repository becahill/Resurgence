use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::cmp::Ordering;
use thiserror::Error;

#[derive(Debug, Error)]
enum CoreError {
    #[error("expected at least 2 return observations")]
    InsufficientReturns,
    #[error("confidence_level must be between 0 and 1")]
    InvalidConfidence,
    #[error("simulations must be greater than 0")]
    InvalidSimulations,
    #[error("horizon_days must be greater than 0")]
    InvalidHorizon,
    #[error("return volatility is effectively zero")]
    DegenerateVolatility,
}

impl From<CoreError> for PyErr {
    fn from(value: CoreError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct SimulationResult {
    #[pyo3(get)]
    pub var: f64,
    #[pyo3(get)]
    pub cvar: f64,
    #[pyo3(get)]
    pub mean_loss: f64,
    #[pyo3(get)]
    pub loss_stddev: f64,
    #[pyo3(get)]
    pub confidence_level: f64,
    #[pyo3(get)]
    pub simulations: usize,
    #[pyo3(get)]
    pub horizon_days: usize,
    #[pyo3(get)]
    pub estimated_drift: f64,
    #[pyo3(get)]
    pub estimated_volatility: f64,
}

#[pymethods]
impl SimulationResult {
    fn __repr__(&self) -> String {
        format!(
            "SimulationResult(var={:.6}, cvar={:.6}, confidence_level={:.4}, simulations={}, horizon_days={})",
            self.var, self.cvar, self.confidence_level, self.simulations, self.horizon_days
        )
    }
}

fn mean_stddev(data: &[f64]) -> Result<(f64, f64), CoreError> {
    if data.len() < 2 {
        return Err(CoreError::InsufficientReturns);
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (n - 1.0);

    let stddev = variance.sqrt();
    Ok((mean, stddev))
}

fn validate_inputs(
    returns: &[f64],
    confidence_level: f64,
    simulations: usize,
    horizon_days: usize,
) -> Result<(), CoreError> {
    if returns.len() < 2 {
        return Err(CoreError::InsufficientReturns);
    }
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(CoreError::InvalidConfidence);
    }
    if simulations == 0 {
        return Err(CoreError::InvalidSimulations);
    }
    if horizon_days == 0 {
        return Err(CoreError::InvalidHorizon);
    }

    Ok(())
}

#[pyfunction]
#[pyo3(signature=(returns, confidence_level=0.95, simulations=10_000, horizon_days=10, seed=None))]
pub fn simulate_var_cvar(
    returns: Vec<f64>,
    confidence_level: f64,
    simulations: usize,
    horizon_days: usize,
    seed: Option<u64>,
) -> PyResult<SimulationResult> {
    validate_inputs(&returns, confidence_level, simulations, horizon_days)?;

    let (drift, volatility) = mean_stddev(&returns)?;
    if volatility <= f64::EPSILON {
        return Err(CoreError::DegenerateVolatility.into());
    }

    let normal = Normal::new(0.0, 1.0).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let base_seed = seed.unwrap_or(42);

    let mut losses = (0..simulations)
        .into_par_iter()
        .map(|i| {
            // Mix the base seed per path for reproducible but independent streams.
            let stream_seed = base_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut rng = StdRng::seed_from_u64(stream_seed);
            let mut portfolio_value = 1.0_f64;

            for _ in 0..horizon_days {
                let z = normal.sample(&mut rng);
                let log_return = drift - 0.5 * volatility * volatility + volatility * z;
                portfolio_value *= log_return.exp();
            }

            let pnl = portfolio_value - 1.0;
            (-pnl).max(0.0)
        })
        .collect::<Vec<f64>>();

    losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let var_index = ((confidence_level * simulations as f64).ceil() as usize)
        .saturating_sub(1)
        .min(simulations - 1);

    let var = losses[var_index];
    let tail_losses = &losses[var_index..];
    let cvar = tail_losses.iter().sum::<f64>() / tail_losses.len() as f64;

    let mean_loss = losses.iter().sum::<f64>() / simulations as f64;
    let loss_variance = losses
        .iter()
        .map(|loss| {
            let delta = loss - mean_loss;
            delta * delta
        })
        .sum::<f64>()
        / (simulations.saturating_sub(1).max(1) as f64);

    Ok(SimulationResult {
        var,
        cvar,
        mean_loss,
        loss_stddev: loss_variance.sqrt(),
        confidence_level,
        simulations,
        horizon_days,
        estimated_drift: drift,
        estimated_volatility: volatility,
    })
}

#[pymodule]
fn resurgence_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SimulationResult>()?;
    module.add_function(wrap_pyfunction!(simulate_var_cvar, module)?)?;
    Ok(())
}
