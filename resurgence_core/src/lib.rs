use ndarray::Array1;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
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
    #[error("initial_state must be in [0, 2]")]
    InvalidInitialState,
    #[error("transition_matrix must be 3x3, non-negative, and each row must sum to 1.0")]
    InvalidTransitionMatrix,
    #[error("{0} must contain exactly 3 values")]
    InvalidStateVector(String),
    #[error("state_vol_multipliers must be strictly positive")]
    InvalidVolatilityMultipliers,
    #[error("rayon_threads must be > 0")]
    InvalidRayonThreads,
    #[error("failed to initialize custom rayon thread pool")]
    ThreadPoolInitFailure,
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
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub max_loss_zscore: f64,
    #[pyo3(get)]
    pub state_occupancy: Vec<f64>,
}

#[pymethods]
impl SimulationResult {
    fn __repr__(&self) -> String {
        format!(
            "SimulationResult(var={:.6}, cvar={:.6}, confidence_level={:.4}, simulations={}, horizon_days={}, sharpe={:.4}, max_loss_zscore={:.4})",
            self.var,
            self.cvar,
            self.confidence_level,
            self.simulations,
            self.horizon_days,
            self.sharpe_ratio,
            self.max_loss_zscore
        )
    }
}

fn default_transition_matrix() -> [[f64; 3]; 3] {
    [
        [0.92, 0.04, 0.04],
        [0.10, 0.80, 0.10],
        [0.18, 0.14, 0.68],
    ]
}

fn default_state_vol_multipliers() -> [f64; 3] {
    [0.70, 1.85, 1.00]
}

fn default_state_drift_adjustments() -> [f64; 3] {
    [0.0003, -0.0008, 0.0]
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
    initial_state: usize,
    rayon_threads: Option<usize>,
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
    if initial_state > 2 {
        return Err(CoreError::InvalidInitialState);
    }
    if let Some(threads) = rayon_threads {
        if threads == 0 {
            return Err(CoreError::InvalidRayonThreads);
        }
    }

    Ok(())
}

fn parse_transition_matrix(raw: Vec<Vec<f64>>) -> Result<[[f64; 3]; 3], CoreError> {
    if raw.len() != 3 {
        return Err(CoreError::InvalidTransitionMatrix);
    }

    let mut matrix = [[0.0_f64; 3]; 3];
    for (row_idx, row) in raw.iter().enumerate() {
        if row.len() != 3 {
            return Err(CoreError::InvalidTransitionMatrix);
        }
        let mut row_sum = 0.0;
        for (col_idx, value) in row.iter().enumerate() {
            if *value < 0.0 || !value.is_finite() {
                return Err(CoreError::InvalidTransitionMatrix);
            }
            matrix[row_idx][col_idx] = *value;
            row_sum += *value;
        }
        if (row_sum - 1.0).abs() > 1e-6 {
            return Err(CoreError::InvalidTransitionMatrix);
        }
    }

    Ok(matrix)
}

fn parse_state_vector(raw: Vec<f64>, label: &str) -> Result<[f64; 3], CoreError> {
    if raw.len() != 3 {
        return Err(CoreError::InvalidStateVector(label.to_string()));
    }

    Ok([raw[0], raw[1], raw[2]])
}

fn sample_next_state(current_state: usize, transition_matrix: &[[f64; 3]; 3], draw: f64) -> usize {
    let mut cumulative = 0.0;
    for (next_state, probability) in transition_matrix[current_state].iter().enumerate() {
        cumulative += *probability;
        if draw <= cumulative || next_state == 2 {
            return next_state;
        }
    }
    2
}

fn simulate_paths(
    simulations: usize,
    horizon_days: usize,
    drift: f64,
    base_volatility: f64,
    transition_matrix: [[f64; 3]; 3],
    state_vol_multipliers: [f64; 3],
    state_drift_adjustments: [f64; 3],
    initial_state: usize,
    seed: u64,
    normal: &Normal<f64>,
    rayon_threads: Option<usize>,
) -> Result<Vec<(f64, f64, [usize; 3])>, CoreError> {
    let run = || {
        (0..simulations)
            .into_par_iter()
            .map(|i| {
                let stream_seed = seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut rng = StdRng::seed_from_u64(stream_seed);

                let mut state = initial_state;
                let mut portfolio_value = 1.0_f64;
                let mut state_counts = [0_usize; 3];

                for _ in 0..horizon_days {
                    let transition_draw: f64 = rng.gen();
                    state = sample_next_state(state, &transition_matrix, transition_draw);
                    state_counts[state] += 1;

                    let sigma =
                        (base_volatility * state_vol_multipliers[state]).max(f64::EPSILON);
                    let mu = drift + state_drift_adjustments[state];
                    let z = normal.sample(&mut rng);

                    let log_return = mu - 0.5 * sigma * sigma + sigma * z;
                    portfolio_value *= log_return.exp();
                }

                let pnl = portfolio_value - 1.0;
                let loss = (-pnl).max(0.0);
                (loss, pnl, state_counts)
            })
            .collect::<Vec<(f64, f64, [usize; 3])>>()
    };

    if let Some(thread_count) = rayon_threads {
        let pool = ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|_| CoreError::ThreadPoolInitFailure)?;
        Ok(pool.install(run))
    } else {
        Ok(run())
    }
}

fn run_regime_monte_carlo(
    returns: &[f64],
    confidence_level: f64,
    simulations: usize,
    horizon_days: usize,
    transition_matrix: [[f64; 3]; 3],
    state_vol_multipliers: [f64; 3],
    state_drift_adjustments: [f64; 3],
    initial_state: usize,
    seed: Option<u64>,
    rayon_threads: Option<usize>,
) -> Result<(SimulationResult, Vec<f64>), CoreError> {
    validate_inputs(
        returns,
        confidence_level,
        simulations,
        horizon_days,
        initial_state,
        rayon_threads,
    )?;

    if state_vol_multipliers.iter().any(|value| *value <= 0.0) {
        return Err(CoreError::InvalidVolatilityMultipliers);
    }

    let (drift, volatility) = mean_stddev(returns)?;
    if volatility <= f64::EPSILON {
        return Err(CoreError::DegenerateVolatility);
    }

    let normal = Normal::new(0.0, 1.0).map_err(|_| CoreError::DegenerateVolatility)?;
    let base_seed = seed.unwrap_or(42);

    let path_outputs = simulate_paths(
        simulations,
        horizon_days,
        drift,
        volatility,
        transition_matrix,
        state_vol_multipliers,
        state_drift_adjustments,
        initial_state,
        base_seed,
        &normal,
        rayon_threads,
    )?;

    let mut losses = Vec::with_capacity(simulations);
    let mut pnls = Vec::with_capacity(simulations);
    let mut state_counts = [0_usize; 3];

    for (loss, pnl, per_path_state_counts) in path_outputs {
        losses.push(loss);
        pnls.push(pnl);
        for idx in 0..3 {
            state_counts[idx] += per_path_state_counts[idx];
        }
    }

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
    let loss_stddev = loss_variance.sqrt();

    let mean_pnl = pnls.iter().sum::<f64>() / simulations as f64;
    let pnl_variance = pnls
        .iter()
        .map(|pnl| {
            let delta = pnl - mean_pnl;
            delta * delta
        })
        .sum::<f64>()
        / (simulations.saturating_sub(1).max(1) as f64);
    let pnl_stddev = pnl_variance.sqrt();

    let annualization_factor = (252.0 / horizon_days as f64).sqrt();
    let sharpe_ratio = if pnl_stddev <= f64::EPSILON {
        0.0
    } else {
        (mean_pnl / pnl_stddev) * annualization_factor
    };

    let max_loss = *losses.last().unwrap_or(&0.0);
    let max_loss_zscore = if loss_stddev <= f64::EPSILON {
        0.0
    } else {
        (max_loss - mean_loss) / loss_stddev
    };

    let total_state_steps = (simulations * horizon_days) as f64;
    let state_occupancy = if total_state_steps <= 0.0 {
        vec![0.0, 0.0, 0.0]
    } else {
        state_counts
            .iter()
            .map(|count| (*count as f64) / total_state_steps)
            .collect::<Vec<f64>>()
    };

    Ok((
        SimulationResult {
            var,
            cvar,
            mean_loss,
            loss_stddev,
            confidence_level,
            simulations,
            horizon_days,
            estimated_drift: drift,
            estimated_volatility: volatility,
            sharpe_ratio,
            max_loss_zscore,
            state_occupancy,
        },
        losses,
    ))
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
    let (result, _) = run_regime_monte_carlo(
        &returns,
        confidence_level,
        simulations,
        horizon_days,
        default_transition_matrix(),
        default_state_vol_multipliers(),
        default_state_drift_adjustments(),
        2,
        seed,
        None,
    )?;

    Ok(result)
}

#[pyfunction]
#[pyo3(signature=(returns, confidence_level=0.95, simulations=10_000, horizon_days=10, transition_matrix=None, state_vol_multipliers=None, state_drift_adjustments=None, initial_state=2, seed=None, rayon_threads=None))]
pub fn simulate_regime_var_cvar<'py>(
    py: Python<'py>,
    returns: Vec<f64>,
    confidence_level: f64,
    simulations: usize,
    horizon_days: usize,
    transition_matrix: Option<Vec<Vec<f64>>>,
    state_vol_multipliers: Option<Vec<f64>>,
    state_drift_adjustments: Option<Vec<f64>>,
    initial_state: usize,
    seed: Option<u64>,
    rayon_threads: Option<usize>,
) -> PyResult<(SimulationResult, Bound<'py, PyArray1<f64>>)> {
    let transition = parse_transition_matrix(
        transition_matrix.unwrap_or_else(|| {
            default_transition_matrix()
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f64>>>()
        }),
    )?;

    let volatility_vector = parse_state_vector(
        state_vol_multipliers.unwrap_or_else(|| default_state_vol_multipliers().to_vec()),
        "state_vol_multipliers",
    )?;

    let drift_vector = parse_state_vector(
        state_drift_adjustments.unwrap_or_else(|| default_state_drift_adjustments().to_vec()),
        "state_drift_adjustments",
    )?;

    let (result, losses) = run_regime_monte_carlo(
        &returns,
        confidence_level,
        simulations,
        horizon_days,
        transition,
        volatility_vector,
        drift_vector,
        initial_state,
        seed,
        rayon_threads,
    )?;

    // Zero-copy handoff: ndarray owns the Rust allocation and NumPy takes ownership
    // directly without cloning Python-side buffers.
    let losses_array = PyArray1::from_owned_array_bound(py, Array1::from_vec(losses));

    Ok((result, losses_array))
}

#[pymodule]
fn resurgence_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SimulationResult>()?;
    module.add_function(wrap_pyfunction!(simulate_var_cvar, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_regime_var_cvar, module)?)?;
    Ok(())
}
