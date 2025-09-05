mod bessel_functions_gsl;
//mod bessel_functions;

use bessel_functions_gsl::{bessel_i1_scaled_fast};
//use bessel_functions::{bessel_i1e_simd};
use interp::{interp, InterpMode};
use ndarray::{s, Array, Array1, ArrayView1, Array2, Array3, ArrayView3, Array4, Axis, Zip};
use rayon::prelude::*;
use std::error::Error;
use std::time::Instant;
use std::iter;

// --- Data Structures for Organization ---

/// Holds all the fixed configuration parameters for the simulation.
struct AppConfig {
    taumin: f64,
    taumax: f64,
    g: f64,
    c: f64,
    lam: f64,
    rd: f64,
    smin: f64,
    smax: f64,
    ds: f64,
    dtau: f64,
    niter: usize,
    adaptive_criterion: f64,
}

/// Holds the computed grids for time, fitness (s), and establishment time (tau).
struct Grids {
    t: Array1<f64>,
    ss: Array1<f64>,
    taus: Array1<f64>,
}

/// Holds pre-computed arrays that don't change during the main iteration.
struct PrecomputedData {
    exps: Array2<f64>,
    logpriorm: Array1<f64>,
    kap: Array1<f64>,
}

/// Holds helper matrices that are re-calculated in each iteration of the main loop.
struct HelperMatrices {
    ee: Array2<f64>,
    eeup: Array1<f64>,
    bkn: Array2<f64>,
    rmu: Array3<f64>,
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
        adaptive_criterion: 0.9f64.ln(),
    }
}

/// Creates and returns the time and parameter grids based on the config.
fn setup_grids(config: &AppConfig) -> Grids {
    let t_values: Vec<f64> = (0..=config.taumax as i32)
        .step_by(config.g as usize)
        .map(|v| v as f64)
        .collect();
    let t = Array1::from_vec(t_values);

    let num_s_steps = ((config.smax - config.smin) / config.ds).round() as i32;
    let ss_values: Vec<f64> = (0..=num_s_steps).map(|i| config.smin + i as f64 * config.ds).collect();
    let ss = Array1::from_vec(ss_values);

    let num_tau_steps = ((config.taumax - config.taumin) / config.dtau).round() as i32;
    let taus_values: Vec<f64> = (0..=num_tau_steps).map(|i| config.taumin + i as f64 * config.dtau).collect();
    let taus = Array1::from_vec(taus_values);

    Grids { t, ss, taus }
}

/// Pre-processes the raw read counts in-place and returns the total reads per time point.
fn preprocess_reads(reads: &mut Array2<f64>) -> Array1<f64> {
    let r: Array1<f64> = reads.sum_axis(Axis(0));
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
    
    let mut exps = Array2::<f64>::zeros((km, ls));
    for k in 1..km {
        for i in 0..ls {
            exps[[k, i]] = (grids.ss[i] * (grids.t[k] - grids.t[k - 1])).exp();
        }
    }

    let big_t = config.taumax - config.taumin;
    let logpriorm = grids.ss.mapv(|s| -s / config.lam + (s / (config.lam.powi(2) * big_t)).ln());

    let kap = Array1::<f64>::from_elem(km, 2.5);

    PrecomputedData { exps, logpriorm, kap }
}

/// Calculates the initial guess for the mean fitness `sbi`.
fn calculate_initial_sbi(reads: &Array2<f64>, grids: &Grids) -> Array1<f64> {
    let n_lin = reads.shape()[0];
    let km = grids.t.len();

    let first_col = reads.column(0);
    let last_col = reads.column(reads.ncols() - 1);
    let mean_first_col = first_col.mean().unwrap_or(0.0);

    let putneut_indices: Vec<usize> = (0..n_lin)
        .filter(|&i| first_col[i] < mean_first_col && first_col[i] > last_col[i] && last_col[i] > 1.0)
        .collect();

    let mut sbi = Array1::<f64>::zeros(km);
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

/// Pre-computes the bk_a values for all lineages, times, fitnesses, and taus.
fn precompute_bk_a(
    reads: &Array2<f64>,
    precomputed: &PrecomputedData,
    helpers: &HelperMatrices,
) -> Array4<f64> {
    let n_lin = reads.shape()[0];
    let km = reads.shape()[1];
    let ls = precomputed.exps.shape()[1];
    let ltau = helpers.rmu.shape()[2];

    let mut bk_a_precomputed = Array4::<f64>::zeros((n_lin, km, ls, ltau));

    bk_a_precomputed.axis_iter_mut(Axis(0))
        .into_iter()
        .enumerate()
        .for_each(|(l, mut bk_a_slice_l)| {
            let mut k0_slice = bk_a_slice_l.slice_mut(s![0, .., ..]);
            k0_slice.fill(reads[[l, 0]]);
            // ------------------------------------------

            // Now, compute the rest for k > 0
            for k in 1..km {
                for i in 0..ls {
                    for j in 0..ltau {
                        let reads_lk_minus_1 = reads[[l, k - 1]];
                        let rmu_val = helpers.rmu[[k, i, j]];
                        
                        let val = (reads_lk_minus_1 - (1.0 - precomputed.exps[[k, i]]) * rmu_val.min(reads_lk_minus_1)) 
                                  * helpers.eeup[k];

                        bk_a_slice_l[[k, i, j]] = val;
                    }
                }
            }
        });

    bk_a_precomputed
}


// --- Main Algorithm Functions ---

/// Updates the set of helper matrices based on the current mean fitness `sb`.
fn update_helper_matrices(
    sb_x: &[f64],
    sb_y: &[f64],
    reads: &Array2<f64>,
    grids: &Grids,
    config: &AppConfig,
) -> HelperMatrices {
    let interp_mode = InterpMode::default();
    let km = grids.t.len();
    let ls = grids.ss.len();
    let ltau = grids.taus.len();

    // Calculate ci
    let t_end = grids.t[km - 1];
    let ci_vec: Vec<f64> = (0..=t_end as i32)
        .map(|i| {
            let upper_bound = i as f64;
            let n_steps = ((upper_bound * 1000.0).ceil() as usize).max(100);
            trapezoidal_rule(|x| interp(sb_x, sb_y, x, &interp_mode), 0.0, upper_bound, n_steps)
        })
        .collect();
    let ci = Array1::from_vec(ci_vec);

    // Calculate ee
    let mut ee = Array2::<f64>::zeros((ci.len(), ci.len()));
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
    let mut eer = Array2::<f64>::zeros((km, km));
    for k in 0..km { eer[[k, k]] = 1.0; }
    for k in 1..km { eer[[k - 1, k]] = eeup[k]; }
    let bkn = reads.dot(&eer);

    // Calculate rmu
    let mut rmu = Array3::<f64>::zeros((km, ls, ltau));
    for k in 1..km {
        let tk_minus_1 = grids.t[k - 1];
        for i in 0..ls {
            let s = grids.ss[i];
            for j in 0..ltau {
                let tau = grids.taus[j];
                if tk_minus_1 >= tau {
                    let sb_tau = interp(sb_x, sb_y, tau, &interp_mode);
                    let denominator = (s - sb_tau).max(0.005);
                    let ee_idx_tau = tau.max(0.0) as usize;
                    let ee_val = ee[[ee_idx_tau, tk_minus_1 as usize]];
                    rmu[[k, i, j]] = (config.rd / config.g) * config.c / denominator * ee_val * (s * (tk_minus_1 - tau)).exp();
                }
            }
        }
    }
    
    HelperMatrices { ee, eeup, bkn, rmu }
}

thread_local! {
    static BK_SCRATCH_BUFFER: std::cell::RefCell<Vec<f64>> = std::cell::RefCell::new(Vec::new());
}

/// Estimates the fitness `s` and establishment time `tau` for each lineage in parallel.
fn estimate_mutations_for_lineages(
    reads: &Array2<f64>,
    grids: &Grids,
    config: &AppConfig,
    precomputed: &PrecomputedData,
    helpers: &HelperMatrices,
    bk_a_precomputed: &Array4<f64>,
) -> Array2<f64> {
    let n_lin = reads.shape()[0];
    let ls = grids.ss.len();
    let ltau = grids.taus.len();

    let muti_rows: Vec<(f64, f64)> = (0..n_lin)
        .into_par_iter()
        .map(|l| {
            let mut log_f = Array2::<f64>::zeros((ls, ltau));
            let log_f = logp_a_vectorizd_all(l, reads, &precomputed.kap, &bk_a_precomputed, &precomputed.logpriorm);
            
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

    let mut muti = Array2::<f64>::zeros((n_lin, 2));
    for (l, (s_val, tau_val)) in muti_rows.into_iter().enumerate() {
        muti[[l, 0]] = s_val;
        muti[[l, 1]] = tau_val;
    }
    muti
}

/// Updates the mean fitness `sbi` based on the latest mutation estimates.
fn update_mean_fitness(
    muti: &Array2<f64>,
    sb_x: &[f64],
    sb_y: &[f64],
    reads: &Array2<f64>,
    r: &Array1<f64>,
    grids: &Grids,
    config: &AppConfig,
    helpers: &HelperMatrices,
) -> Array1<f64> {
    let n_lin = reads.shape()[0];
    let km = grids.t.len();
    let interp_mode = InterpMode::default();

    let mut new_sbi = Array1::zeros(km);
    for k in 1..km {
        let sbi_k_numerator: f64 = (0..n_lin).into_par_iter().map(|l| {
            let s = muti[[l, 0]];
            let tau = muti[[l, 1]];
            if s > 0.0 {
                let sb_tau = interp(sb_x, sb_y, tau, &interp_mode);
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
    let mut sbmat = Array2::<f64>::zeros((config.niter, grids.t.len()));
    let mut sb_x: Vec<f64> = std::iter::once(config.taumin).chain(grids.t.iter().cloned()).collect();
    let mut sb_y: Vec<f64> = std::iter::once(0.0).chain(initial_sbi.iter().cloned()).collect();

    for iter in 0..config.niter {
        let t1 = Instant::now();
        println!("Iteration: {}", iter + 1);

        // --- 2a. Update helper matrices based on current `sb` ---
        let helpers = update_helper_matrices(&sb_x, &sb_y, &reads, &grids, &config);

        // --- 2b. Precompute bk_a values ---
        let bk_a_precomputed = precompute_bk_a(&reads, &precomputed, &helpers);

        // --- 2c. Estimate `s` and `tau` for each lineage ---
        let muti = estimate_mutations_for_lineages(&reads, &grids, &config, &precomputed, &helpers, &bk_a_precomputed);

        // --- 2d. Update mean fitness `sb` using new estimates ---
        let new_sbi = update_mean_fitness(&muti, &sb_x, &sb_y, &reads, &r, &grids, &config, &helpers);
        
        // --- 2e. Store results and update state for next iteration ---
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

fn read_csv_to_ndarray(filepath: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(filepath)?;
    let mut records = Vec::new();
    let mut ncols = 0;
    for result in rdr.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().map(|s| s.parse().unwrap()).collect();
        if ncols == 0 { ncols = row.len(); }
        records.extend_from_slice(&row);
    }
    let nrows = records.len() / ncols;
    Ok(Array::from_shape_vec((nrows, ncols), records)?)
}

fn logsumexp<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let vec: Vec<f64> = iter.collect();
    if vec.is_empty() { return f64::NEG_INFINITY; }
    let max_val = vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if max_val.is_infinite() { return max_val; }
    let sum = vec.iter().map(|&x| (x - max_val).exp()).sum::<f64>();
    max_val + sum.ln()
}

fn argmax_2d(arr: &Array2<f64>) -> (usize, usize) {
    arr.indexed_iter()
       .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
       .map(|(index, _)| index) 
       .unwrap_or((0, 0))
}

fn median(data: &mut [f64]) -> f64 {
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    }
}

fn trapezoidal_rule<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    if (a - b).abs() < 1e-9 { return 0.0; }
    let h = (b - a) / (n as f64);
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        let x = a + (i as f64) * h;
        sum += f(x);
    }
    sum * h
}

fn bk_a(l: usize, k: usize, i: usize, j: usize, reads: &Array2<f64>, exps: &Array2<f64>, rmu: &Array3<f64>, eeup: &Array1<f64>) -> f64 {
    let reads_lk_minus_1 = reads[[l, k - 1]];
    let rmu_val = rmu[[k, i, j]];
    (reads_lk_minus_1 - (1.0 - exps[[k, i]]) * rmu_val.min(reads_lk_minus_1)) * eeup[k]
}


fn logexpi(r: f64, bk: f64, kappa: f64) -> f64 {

    let x = 2.0 * (r * bk).sqrt() / kappa;

    let bessel_term = bessel_i1_scaled_fast(x as f64) as f64;

    -(kappa * (1.0 - (-bk / kappa).exp())).ln() + 0.5 * (bk / r).ln() - (r + bk) / kappa + bessel_term.ln() + x
}

fn logexpi_vectorized(r_vals: ArrayView1<f64>, bk_vals: ArrayView1<f64>, kaps: ArrayView1<f64>) -> f64 {
    
    let xs = 2.0 * ((&r_vals * &bk_vals).sqrt() / &kaps);

    // if I can vectorize this with SIMD, then we're cruising.
    //let bessel_terms: Vec<f64> = xs.clone().into_iter().map(bessel_i1_scaled_fast).collect();
    let bessel_terms = xs.mapv(bessel_i1_scaled_fast);

    let mut sum = 0.0;

    // Zip up all the input arrays needed for the calculation.
    Zip::from(&r_vals)
        .and(&bk_vals)
        .and(&kaps)
        .and(&bessel_terms)
        .and(&xs)
        // .for_each() executes a closure for each element set. It returns ().
        .for_each(|&r, &bk, &kappa, &bes, &x| {
            // This is the body of our single, fused loop.
            // Calculate the term for the current set of elements...
            let term = -(kappa * (1.0 - (-bk / kappa).exp())).ln()
                     + 0.5 * (bk / r).ln()
                     - (r + bk) / kappa
                     + bes.ln()
                     + x;
            // ...and add it to our accumulator.
            sum += term;
        });

    sum

}

fn logexpi_vectorized_all(r_vals: ArrayView1<f64>, bk_vals: ArrayView3<f64>, kaps: ArrayView1<f64>) -> Array2<f64> {
    
    let ls = bk_vals.shape()[1];
    let ltau = bk_vals.shape()[2];

    // The only allocation. This will be our `sum` array.
    let mut log_likelihood_2d = Array2::<f64>::zeros((ls, ltau));

    // Outer loop over 'k'.
    for k in 0..r_vals.len() {
        let r_k = r_vals[k];
        let kap_k = kaps[k];
        let bk_slice_2d = bk_vals.slice(s![k, .., ..]);

        // This Zip iterates over the (i, j) dimensions.
        Zip::from(&mut log_likelihood_2d)
            .and(&bk_slice_2d)
            .for_each(|ll_val, &bk_val| {
                // --- Perform the calculation directly ---
                // This might produce NaN or -inf if inputs are zero or negative.
                let x = 2.0 * (r_k * bk_val).sqrt() / kap_k;
                let bessel_term = bessel_i1_scaled_fast(x);

                let term = -(kap_k * (1.0 - (-bk_val / kap_k).exp())).ln()
                           + 0.5 * (bk_val / r_k).ln()
                           - (r_k + bk_val) / kap_k
                           + bessel_term.ln()
                           + x;
                           
                *ll_val +=  term;
            });
    }
    
    log_likelihood_2d // This is the final summed array
}

fn logp_n(l: usize, reads: &Array2<f64>, bkn: &Array2<f64>, kap: &Array1<f64>) -> f64 {
    let km = reads.shape()[1];
    (0..km).map(|k| {
        let r_val = reads[[l, k]];
        let bk_val = if k == 0 { r_val } else { bkn[[l, k]] };
        logexpi(r_val, bk_val, kap[k])
    }).sum()
}

/*
fn logp_a(l: usize, i: usize, j: usize, reads: &Array2<f64>, kap: &Array1<f64>, exps: &Array2<f64>, rmu: &Array3<f64>, eeup: &Array1<f64>, logpriorm: &Array1<f64>) -> f64 {
    let km = reads.shape()[1];
    let mut logp = logpriorm[i];
    for k in 0..km {
        let r_val = reads[[l, k]];
        let bk_val = if k == 0 { r_val } else { bk_a(l, k, i, j, reads, exps, rmu, eeup) };
        logp += logexpi(r_val, bk_val, kap[k]);
    }
    logp
}
*/

fn logp_a_vectorizd(
    l: usize, 
    i: usize, 
    j: usize, 
    reads: &Array2<f64>, 
    kap: &Array1<f64>, 
    bk_a_precomputed: &Array4<f64>, 
    logpriorm: &Array1<f64>
) -> f64 {
    let mut logp = logpriorm[i];
    let r_vals_view = reads.row(l);

    // Just take a single, clean slice from the precomputed array.
    // This slice represents all `k` values for the given l, i, and j.
    // This is a view, so it's zero-cost.
    let bk_vals_view = bk_a_precomputed.slice(s![l, .., i, j]);

    // Pass the views directly.
    logp += logexpi_vectorized(r_vals_view, bk_vals_view, kap.view());

    logp
}

fn logp_a_vectorizd_all(
    l: usize, 
    reads: &Array2<f64>, 
    kap: &Array1<f64>, 
    bk_a_precomputed: &Array4<f64>, 
    logpriorm: &Array1<f64>
) -> Array2<f64> {

    let r_vals_view = reads.row(l);

    // Just take a single, clean slice from the precomputed array.
    // This slice represents all `k` values for the given l, i, and j.
    // This is a view, so it's zero-cost.
    let bk_vals_view = bk_a_precomputed.slice(s![l, .., .., ..]);

    // Pass the views directly.
    let log_liklihood = logexpi_vectorized_all(r_vals_view, bk_vals_view, kap.view());
    let log_f = log_liklihood + &logpriorm.clone().insert_axis(Axis(1));

    log_f
}
