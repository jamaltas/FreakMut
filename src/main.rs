mod bessel_functions_gsl;
//mod bessel_functions;

mod interp_zero_alloc;

use bessel_functions_gsl::{bessel_i1_scaled_fast};
//use bessel_functions::{bessel_i1e_simd};

use interp_zero_alloc::{interp_zero_alloc, InterpMode};

//use interp::{interp, InterpMode};
use ndarray::{Array, Array1, ArrayView1, Array2, Array3, Axis, Zip};
use rayon::prelude::*;
use std::error::Error;
use std::time::Instant;
use std::iter;
use fastapprox::faster;

// --- Data Structures for Organization ---

/// Holds all the fixed configuration parameters for the simulation.
struct AppConfig {
    taumin: f32,
    taumax: f32,
    g: f32,
    c: f32,
    lam: f32,
    rd: f32,
    smin: f32,
    smax: f32,
    ds: f32,
    dtau: f32,
    niter: usize,
    adaptive_criterion: f32,
}

/// Holds the computed grids for time, fitness (s), and establishment time (tau).
struct Grids {
    t: Array1<f32>,
    ss: Array1<f32>,
    taus: Array1<f32>,
}

/// Holds pre-computed arrays that don't change during the main iteration.
struct PrecomputedData {
    exps: Array2<f32>,
    logpriorm: Array1<f32>,
    kap: Array1<f32>,
}

/// Holds helper matrices that are re-calculated in each iteration of the main loop.
struct HelperMatrices {
    ee: Array2<f32>,
    eeup: Array1<f32>,
    bkn: Array2<f32>,
    rmu: Array3<f32>,
}


// --- Setup and Initialization Functions ---

/// Creates and returns the application's configuration parameters.
fn setup_config() -> AppConfig {
    AppConfig {
        taumin: -20.0,
        taumax: 112.0,
        g: 8.0,
        c: 2.0,
        lam: 0.1,
        rd: 1.0,
        smin: 0.01,
        smax: 0.25,
        ds: 0.01,
        dtau: 4.0,
        niter: 5,
        adaptive_criterion: 0.9f32.ln(),
    }
}

/// Creates and returns the time and parameter grids based on the config.
fn setup_grids(config: &AppConfig) -> Grids {
    let t_values: Vec<f32> = (0..=config.taumax as i32)
        .step_by(config.g as usize)
        .map(|v| v as f32)
        .collect();
    let t = Array1::from_vec(t_values);

    let num_s_steps = ((config.smax - config.smin) / config.ds).round() as i32;
    let ss_values: Vec<f32> = (0..=num_s_steps).map(|i| config.smin + i as f32 * config.ds).collect();
    let ss = Array1::from_vec(ss_values);

    let num_tau_steps = ((config.taumax - config.taumin) / config.dtau).round() as i32;
    let taus_values: Vec<f32> = (0..=num_tau_steps).map(|i| config.taumin + i as f32 * config.dtau).collect();
    let taus = Array1::from_vec(taus_values);

    Grids { t, ss, taus }
}

/// Pre-processes the raw read counts in-place and returns the total reads per time point.
fn preprocess_reads(reads: &mut Array2<f32>) -> Array1<f32> {
    let r: Array1<f32> = reads.sum_axis(Axis(0));
    let r1_fix_factor = r[1] / r[0];
    
    let mut reads_col0 = reads.column_mut(0);
    reads_col0.mapv_inplace(|x| (r1_fix_factor * x).round());
    reads.mapv_inplace(|x| if x < 1.0 { 1.0 } else { x });
    
    r
}

/// Pre-computes arrays that are constant throughout the iterations.
fn precompute_data(grids: &Grids, config: &AppConfig) -> PrecomputedData {
    let km = grids.t.len();
    let ls = grids.ss.len();
    
    let mut exps = Array2::<f32>::zeros((km, ls));
    for k in 1..km {
        for i in 0..ls {
            exps[[k, i]] = (grids.ss[i] * (grids.t[k] - grids.t[k - 1])).exp();
        }
    }

    let big_t = config.taumax - config.taumin;
    let logpriorm = grids.ss.mapv(|s| -s / config.lam + (s / (config.lam.powi(2) * big_t)).ln());

    let kap = Array1::<f32>::from_elem(km, 2.5);

    PrecomputedData { exps, logpriorm, kap }
}

/// Calculates the initial guess for the mean fitness `sbi`.
fn calculate_initial_sbi(reads: &Array2<f32>, grids: &Grids) -> Array1<f32> {
    let n_lin = reads.shape()[0];
    let km = grids.t.len();

    let first_col = reads.column(0);
    let last_col = reads.column(reads.ncols() - 1);
    let mean_first_col = first_col.mean().unwrap_or(0.0);

    let putneut_indices: Vec<usize> = (0..n_lin)
        .filter(|&i| first_col[i] < mean_first_col && first_col[i] > last_col[i] && last_col[i] > 1.0)
        .collect();

    let mut sbi = Array1::<f32>::zeros(km);
    if !putneut_indices.is_empty() {
        let putneut_reads = reads.select(Axis(0), &putneut_indices);
        for k in 1..km {
            let mut col_k_data = putneut_reads.column(k).to_vec();
            let mut col_k_minus_1_data = putneut_reads.column(k - 1).to_vec();
            let median_k = median(&mut col_k_data);
            let median_k_minus_1 = median(&mut col_k_minus_1_data);
            
            let s_val = if median_k > 0.0 && median_k_minus_1 > 0.0 {
                -(median_k / median_k_minus_1).ln() / (grids.t[k] - grids.t[k - 1])
            } else {
                0.0
            };
            sbi[k] = s_val.max(sbi[k - 1]);
        }
    }
    sbi
}


// --- Main Algorithm Functions ---

/// Updates the set of helper matrices based on the current mean fitness `sb`.
fn update_helper_matrices(
    sb_x: &[f32],
    sb_y: &[f32],
    reads: &Array2<f32>,
    grids: &Grids,
    config: &AppConfig,
) -> HelperMatrices {
    let interp_mode = InterpMode::Extrapolate;
    let km = grids.t.len();
    let ls = grids.ss.len();
    let ltau = grids.taus.len();

    // Calculate ci
    let t_end = grids.t[km - 1];
    let ci_vec: Vec<f32> = (0..=t_end as i32)
        .map(|i| {
            let upper_bound = i as f32;
            let n_steps = ((upper_bound * 1000.0).ceil() as usize).max(100);
            trapezoidal_rule(|x| interp_zero_alloc(sb_x, sb_y, x, &interp_mode), 0.0, upper_bound, n_steps)
        })
        .collect();
    let ci = Array1::from_vec(ci_vec);

    // Calculate ee
    let mut ee = Array2::<f32>::zeros((ci.len(), ci.len()));
    for i in 0..ci.len() {
        for j in 0..ci.len() {
            ee[[i, j]] = (ci[i] - ci[j]).exp();
        }
    }

    // Calculate eeup
    let mut eeup_vec = vec![1.0];
    eeup_vec.extend((1..km).map(|k| ee[[grids.t[k - 1] as usize, grids.t[k] as usize]]));
    let eeup = Array1::from_vec(eeup_vec);

    // Calculate bkn
    let mut eer = Array2::<f32>::zeros((km, km));
    for k in 0..km { eer[[k, k]] = 1.0; }
    for k in 1..km { eer[[k - 1, k]] = eeup[k]; }
    let bkn = reads.dot(&eer);

    // Calculate rmu
    let mut rmu = Array3::<f32>::zeros((ls, ltau, km));
    for i in 0..ls {
        let s = grids.ss[i];
        for j in 0..ltau {
            let tau = grids.taus[j];
            let sb_tau = interp_zero_alloc(sb_x, sb_y, tau, &interp_mode);
            let denominator = (s - sb_tau).max(0.005);
            let ee_idx_tau = tau.max(0.0) as usize;

            for k in 1..km {
                let tk_minus_1 = grids.t[k - 1];
                if tk_minus_1 >= tau {
                    let ee_val = ee[[ee_idx_tau, tk_minus_1 as usize]];
                    // Note the new index order [i, j, k]
                    rmu[[i, j, k]] = (config.rd / config.g) * config.c / denominator * ee_val * (s * (tk_minus_1 - tau)).exp();
                }
            }
        }
    }
    
    HelperMatrices { ee, eeup, bkn, rmu }
}

/// Estimates the fitness `s` and establishment time `tau` for each lineage in parallel.
fn estimate_mutations_for_lineages(
    reads: &Array2<f32>,
    grids: &Grids,
    config: &AppConfig,
    precomputed: &PrecomputedData,
    helpers: &HelperMatrices,
) -> Array2<f32> {
    let n_lin = reads.shape()[0];
    let ls = grids.ss.len();
    let ltau = grids.taus.len();

    let muti_rows: Vec<(f32, f32)> = (0..n_lin)
        .into_par_iter()
        //.into_iter()
        .map(|l| {
            let mut log_f = Array2::<f32>::zeros((ls, ltau));
            for i in 0..ls {
                for j in 0..ltau {
                    log_f[[i, j]] = logp_a(l, i, j, reads, &precomputed.kap, &precomputed.exps, &helpers.rmu, &helpers.eeup, &precomputed.logpriorm);
                    //log_f[[i, j]] = logp_a_vectorizd(l, i, j, reads, &precomputed.kap, &precomputed.exps, &helpers.rmu, &helpers.eeup, &precomputed.logpriorm);
                }
            }
            
            let pra = logsumexp(log_f.iter().cloned()) + (config.ds * config.dtau).ln();
            let prn = logp_n(l, reads, &helpers.bkn, &precomputed.kap);
            
            let log_posterior_adaptive = pra - logsumexp([pra, prn].into_iter());

            if log_posterior_adaptive > config.adaptive_criterion {
                let (max_i, max_j) = argmax_2d(&log_f);
                (grids.ss[max_i], grids.taus[max_j])
            } else {
                (0.0, 0.0)
            }
        })
        .collect();

    let mut muti = Array2::<f32>::zeros((n_lin, 2));
    for (l, (s_val, tau_val)) in muti_rows.into_iter().enumerate() {
        muti[[l, 0]] = s_val;
        muti[[l, 1]] = tau_val;
    }
    muti
}

/// Updates the mean fitness `sbi` based on the latest mutation estimates.
fn update_mean_fitness(
    muti: &Array2<f32>,
    sb_x: &[f32],
    sb_y: &[f32],
    reads: &Array2<f32>,
    r: &Array1<f32>,
    grids: &Grids,
    config: &AppConfig,
    helpers: &HelperMatrices,
) -> Array1<f32> {
    let n_lin = reads.shape()[0];
    let km = grids.t.len();
    let interp_mode = InterpMode::Extrapolate;

    let mut new_sbi = Array1::zeros(km);
    for k in 1..km {
        let sbi_k_numerator: f32 = (0..n_lin).into_par_iter().map(|l| {
            let s = muti[[l, 0]];
            let tau = muti[[l, 1]];
            if s > 0.0 && grids.t[k] > tau {
                let sb_tau = interp_zero_alloc(sb_x, sb_y, tau, &interp_mode);
                let nmut1 = config.c / (s - sb_tau).max(0.005)
                    * helpers.ee[[(tau.max(0.0)) as usize, grids.t[k] as usize]]
                    * (s * (grids.t[k] - tau)).exp();
                let nmut2 = (config.g / config.rd) * reads[[l, k]];
                let nmut = nmut1.min(nmut2);
                nmut * s
            } else {
                0.0
            }
        }).sum();
        new_sbi[k] = sbi_k_numerator / ((config.g / config.rd) * r[k]);
    }
    new_sbi
}


fn main() -> Result<(), Box<dyn Error>> {

    // --- 1. Setup and Pre-computation ---
    let config = setup_config();
    let grids = setup_grids(&config);
    
    let mut reads = read_csv_to_ndarray("no_ecology/simu_3_EvoSimulation_Read_Number.csv")?;
    let r = preprocess_reads(&mut reads);
    
    let precomputed = precompute_data(&grids, &config);
    let initial_sbi = calculate_initial_sbi(&reads, &grids);

    // --- 2. Main Iterative Algorithm ---
    let mut sbmat = Array2::<f32>::zeros((config.niter, grids.t.len()));
    let mut sb_x: Vec<f32> = std::iter::once(config.taumin).chain(grids.t.iter().cloned()).collect();
    let mut sb_y: Vec<f32> = std::iter::once(0.0).chain(initial_sbi.iter().cloned()).collect();

    for iter in 0..config.niter {
        let t1 = Instant::now();
        println!("Iteration: {}", iter + 1);

        // --- 2a. Update helper matrices based on current `sb` ---
        let helpers = update_helper_matrices(&sb_x, &sb_y, &reads, &grids, &config);

        // --- 2b. Estimate `s` and `tau` for each lineage ---
        let muti = estimate_mutations_for_lineages(&reads, &grids, &config, &precomputed, &helpers);

        // --- 2c. Update mean fitness `sb` using new estimates ---
        let new_sbi = update_mean_fitness(&muti, &sb_x, &sb_y, &reads, &r, &grids, &config, &helpers);
        
        // --- 2d. Store results and update state for next iteration ---
        sbmat.row_mut(iter).assign(&new_sbi);
        sb_x = std::iter::once(config.taumin).chain(grids.t.iter().cloned()).collect();
        sb_y = std::iter::once(0.0).chain(new_sbi.iter().cloned()).collect();

        println!("Time for iteration: {:.2?}", t1.elapsed());
    }

    // --- 3. Final Output ---
    println!("\nFinal sbmat:\n{:.4}", sbmat);
    
    Ok(())
}


// --- Helper and Bayesian Functions  ---

fn read_csv_to_ndarray(filepath: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(filepath)?;
    let mut records = Vec::new();
    let mut ncols = 0;
    for result in rdr.records() {
        let record = result?;
        let row: Vec<f32> = record.iter().map(|s| s.parse().unwrap()).collect();
        if ncols == 0 { ncols = row.len(); }
        records.extend_from_slice(&row);
    }
    let nrows = records.len() / ncols;
    Ok(Array::from_shape_vec((nrows, ncols), records)?)
}

fn logsumexp<I: Iterator<Item = f32>>(iter: I) -> f32 {
    let vec: Vec<f32> = iter.collect();
    if vec.is_empty() { return f32::NEG_INFINITY; }
    let max_val = vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val.is_infinite() { return max_val; }
    let sum = vec.iter().map(|&x| (x - max_val).exp()).sum::<f32>();
    max_val + sum.ln()
}

fn argmax_2d(arr: &Array2<f32>) -> (usize, usize) {
    arr.indexed_iter()
       .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
       .map(|(index, _)| index) 
       .unwrap_or((0, 0))
}

fn median(data: &mut [f32]) -> f32 {
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    }
}

fn trapezoidal_rule<F: Fn(f32) -> f32>(f: F, a: f32, b: f32, n: usize) -> f32 {
    if (a - b).abs() < 1e-9 { return 0.0; }
    let h = (b - a) / (n as f32);
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        let x = a + (i as f32) * h;
        sum += f(x);
    }
    sum * h
}

fn bk_a(l: usize, k: usize, i: usize, j: usize, reads: &Array2<f32>, exps: &Array2<f32>, rmu: &Array3<f32>, eeup: &Array1<f32>) -> f32 {
    
    let reads_lk_minus_1 = reads[[l, k - 1]];
    let rmu_val = rmu[[i, j, k]]; 
    (reads_lk_minus_1 - (1.0 - exps[[k, i]]) * rmu_val.min(reads_lk_minus_1)) * eeup[k]
}

#[inline(always)]
fn logexpi(r: f32, bk: f32, kappa: f32) -> f32 {
    let x = 2.0 * (r * bk).sqrt() / kappa;
    let bessel_term = bessel_i1_scaled_fast(x);

    -faster::ln(kappa * (1.0 - faster::exp(-bk / kappa))) 
    + 0.5 * faster::ln(bk / r) 
    - (r + bk) / kappa 
    + faster::ln(bessel_term) 
    + x
}

fn logexpi_vectorized(r_vals: ArrayView1<f32>, bk_vals: ArrayView1<f32>, kaps: ArrayView1<f32>) -> f32 {
    let mut sum = 0.0;

    // Use Zip to iterate over the input arrays only
    Zip::from(&r_vals)
        .and(&bk_vals)
        .and(&kaps)
        // This closure will be executed for each element set in parallel
        .for_each(|&r, &bk, &kappa| {
            // Calculate 'x' for the current element directly
            let x  = 2.0 * ((r * bk).sqrt() / kappa);

            // Calculate 'bes' (bessel_terms) for the current element directly
            let bes = bessel_i1_scaled_fast(x);

            // Calculate the 'term' for the current element
            let term = -faster::ln(kappa * (1.0 - faster::exp(-bk / kappa)))
                     + 0.5 * faster::ln(bk / r)
                     - (r + bk) / kappa
                     + faster::ln(bes)
                     + x;

            // Add to the sum. Rayon handles atomic summing for parallel for_each.
            sum += term;
        });

    sum
}

fn logp_n(l: usize, reads: &Array2<f32>, bkn: &Array2<f32>, kap: &Array1<f32>) -> f32 {
    let km = reads.shape()[1];
    (0..km).map(|k| {
        let r_val = reads[[l, k]];
        let bk_val = if k == 0 { r_val } else { bkn[[l, k]] };
        logexpi(r_val, bk_val, kap[k])
    }).sum()
}


fn logp_a(l: usize, i: usize, j: usize, reads: &Array2<f32>, kap: &Array1<f32>, exps: &Array2<f32>, rmu: &Array3<f32>, eeup: &Array1<f32>, logpriorm: &Array1<f32>) -> f32 {
    let km = reads.shape()[1];
    let mut logp = logpriorm[i];
    let r_vals = reads.row(l);

    // Handle k=0 case
    logp += logexpi(r_vals[0], r_vals[0], kap[0]);
    
    // Loop for k > 0
    for k in 1..km {
        let r_val = r_vals[k];
        let bk_val = bk_a(l, k, i, j, reads, exps, rmu, eeup);
        logp += logexpi(r_val, bk_val, kap[k]);
    }
    logp
}


fn logp_a_vectorizd(l: usize, i: usize, j: usize, reads: &Array2<f32>, kap: &Array1<f32>, exps: &Array2<f32>, rmu: &Array3<f32>, eeup: &Array1<f32>, logpriorm: &Array1<f32>) -> f32 {
    let km = reads.shape()[1];
    let mut logp = logpriorm[i];

    let r_vals_view = reads.row(l);

    // Get the first value
    let bk_first = r_vals_view[0];

    // Create an iterator that yields the first value, then all subsequent values
    let bk_vals_iter = iter::once(bk_first)
        .chain((1..km).map(|k| bk_a(l, k, i, j, reads, exps, rmu, eeup)));

    // Collect the complete iterator directly into a new Array1
    let bk_vals: Array1<f32> = bk_vals_iter.collect();

    logp += logexpi_vectorized(r_vals_view.view(), bk_vals.view(), kap.view());

    logp
}
