//! A Rust translation of GSL's exponentially scaled modified Bessel function
//! of the first kind of order 1.

// GSL constants are available in Rust's standard library.
// We'll use f64's associated constants.

// --- Structs and Error Types ---

/// Represents the result of a GSL special function calculation,
/// containing the value and an estimated absolute error.
/// This is the Rust equivalent of `gsl_sf_result`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GslSfResult {
    pub val: f64,
    pub err: f64,
}

/// Represents the possible errors from GSL functions.
/// This replaces the integer error codes like `GSL_SUCCESS`, `GSL_EUNDRFLW`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GslError {
    Underflow,
    Overflow,
    // Other GSL errors like Domain, NoConverge, etc., could be added here.
}

// Implement the standard Error trait for our custom error type.
impl std::fmt::Display for GslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GslError::Underflow => write!(f, "Underflow error"),
            GslError::Overflow => write!(f, "Overflow error"),
        }
    }
}
impl std::error::Error for GslError {}


/// Represents a Chebyshev series for function approximation.
/// This is the Rust equivalent of `cheb_series_struct`.
/// The pointer `c` is replaced by a safe, static slice `&'static [f64]`.
pub struct ChebSeries {
    /// Coefficients
    pub c: &'static [f64],
    /// Order of expansion
    pub order: usize,
    /// Lower interval point
    pub a: f64,
    /// Upper interval point
    pub b: f64,
    // order_sp is not used in the provided functions, but retained for faithfulness.
    // pub order_sp: usize,
}


// --- Chebyshev Evaluation Function (from cheb_eval.c) ---

/// Evaluates a Chebyshev series.
/// This is a direct translation of the `cheb_eval_e` C function.
/// Instead of taking a mutable pointer for the result, it returns a `GslSfResult`.
/// The C function always returns `GSL_SUCCESS`, so this function doesn't need to return a `Result`.
fn cheb_eval(cs: &ChebSeries, x: f64) -> GslSfResult {
    let mut d = 0.0;
    let mut dd = 0.0;
    let mut e = 0.0;

    let y = (2.0 * x - cs.a - cs.b) / (cs.b - cs.a);
    let y2 = 2.0 * y;

    // The loop in C `for(j = cs->order; j>=1; j--)` is more idiomatically
    // written as a reversed range in Rust.
    for j in (1..=cs.order).rev() {
        let temp = d;
        d = y2 * d - dd + cs.c[j];
        e += (y2 * temp).abs() + dd.abs() + cs.c[j].abs();
        dd = temp;
    }

    // Final term (j=0)
    {
        let temp = d;
        d = y * d - dd + 0.5 * cs.c[0];
        e += (y * temp).abs() + dd.abs() + 0.5 * cs.c[0].abs();
    }

    GslSfResult {
        val: d,
        err: f64::EPSILON * e + cs.c[cs.order].abs(),
    }
}


// --- Static Data for Chebyshev Expansions ---
// Static arrays and structs are used to represent the global data from the C file.
// Rust convention uses UPPER_SNAKE_CASE for static/const items.

// series for bi1 on the interval [0.0, 9.0]
static BI1_DATA: [f64; 11] = [
    -0.001971713261099859,
    0.407348876675464810,
    0.034838994299959456,
    0.001545394556300123,
    0.000041888521098377,
    0.000000764902676483,
    0.000000010042493924,
    0.000000000099322077,
    0.000000000000766380,
    0.000000000000004741,
    0.000000000000000024,
];
static BI1_CS: ChebSeries = ChebSeries {
    c: &BI1_DATA,
    order: 10,
    a: -1.0,
    b: 1.0,
};

// series for ai1 on the interval [0.125, 0.33333]
static AI1_DATA: [f64; 21] = [
    -0.02846744181881479, -0.01922953231443221, -0.00061151858579437,
    -0.00002069971253350, 0.00000858561914581, 0.00000104949824671,
    -0.00000029183389184, -0.00000001559378146, 0.00000001318012367,
    -0.00000000144842341, -0.00000000029085122, 0.00000000012663889,
    -0.00000000001664947, -0.00000000000166665, 0.00000000000124260,
    -0.00000000000027315, 0.00000000000002023, 0.00000000000000730,
    -0.00000000000000333, 0.00000000000000071, -0.00000000000000006,
];
static AI1_CS: ChebSeries = ChebSeries {
    c: &AI1_DATA,
    order: 20,
    a: -1.0,
    b: 1.0,
};

// series for ai12 on the interval [0.0, 0.125]
static AI12_DATA: [f64; 22] = [
    0.02857623501828014, -0.00976109749136147, -0.00011058893876263,
    -0.00000388256480887, -0.00000025122362377, -0.00000002631468847,
    -0.00000000383538039, -0.00000000055897433, -0.00000000001897495,
    0.00000000003252602, 0.00000000001412580, 0.00000000000203564,
    -0.00000000000071985, -0.00000000000040836, -0.00000000000002101,
    0.00000000000004273, 0.00000000000001041, -0.00000000000000382,
    -0.00000000000000186, 0.00000000000000033, 0.00000000000000028,
    -0.00000000000000003,
];
static AI12_CS: ChebSeries = ChebSeries {
    c: &AI12_DATA,
    order: 21,
    a: -1.0,
    b: 1.0,
};


// --- Public Bessel Functions ---

// The C macro `ROOT_EIGHT` becomes a Rust const.
const ROOT_EIGHT: f64 = 2.0 * std::f64::consts::SQRT_2;
// GSL_SQRT_DBL_EPSILON
const EPSILON_SQRT: f64 = 1.4901161193847656e-08; // sqrt(f64::EPSILON)
// GSL_LOG_DBL_MAX
const LOG_DBL_MAX: f64 = 709.782712893384; // f64::MAX.ln()


/// Computes the exponentially scaled modified Bessel function of the first kind of order 1, e^(-|x|) * I_1(x).
///
/// This is a translation of `gsl_sf_bessel_I1_scaled_e`.
pub fn bessel_i1_scaled(x: f64) -> Result<GslSfResult, GslError> {
    // GSL's `GSL_DBL_MIN` is the smallest positive normalized double-precision number.
    // In Rust, this is `f64::MIN_POSITIVE`.
    const X_MIN: f64 = 2.0 * f64::MIN_POSITIVE;
    const X_SMALL: f64 = ROOT_EIGHT * EPSILON_SQRT; // Corrected

    let y = x.abs();

    if y == 0.0 {
        Ok(GslSfResult { val: 0.0, err: 0.0 })
    } else if y < X_MIN {
        // The macro `UNDERFLOW_ERROR(result)` is translated to returning an `Err`.
        Err(GslError::Underflow)
    } else if y < X_SMALL {
        Ok(GslSfResult { val: 0.5 * x, err: 0.0 })
    } else if y <= 3.0 {
        let ey = (-y).exp();
        let c = cheb_eval(&BI1_CS, y * y / 4.5 - 1.0);
        let val = x * ey * (0.875 + c.val);
        let mut err = ey * c.err + y * f64::EPSILON * val.abs();
        err += 2.0 * f64::EPSILON * val.abs();
        Ok(GslSfResult { val, err })
    } else if y <= 8.0 {
        let sy = y.sqrt();
        let c = cheb_eval(&AI1_CS, (48.0 / y - 11.0) / 5.0);
        let b = (0.375 + c.val) / sy;
        // The C `(x > 0.0 ? 1.0 : -1.0)` is perfectly mapped by `f64::signum`.
        let s = x.signum();
        let val = s * b;
        let mut err = c.err / sy;
        err += 2.0 * f64::EPSILON * val.abs();
        Ok(GslSfResult { val, err })
    } else { // y > 8.0
        let sy = y.sqrt();
        let c = cheb_eval(&AI12_CS, 16.0 / y - 1.0);
        let b = (0.375 + c.val) / sy;
        let s = x.signum();
        let val = s * b;
        let mut err = c.err / sy;
        err += 2.0 * f64::EPSILON * val.abs();
        Ok(GslSfResult { val, err })
    }
}


/// Computes the modified Bessel function of the first kind of order 1, I_1(x).
///
/// This is a translation of `gsl_sf_bessel_I1_e`.
pub fn bessel_i1(x: f64) -> Result<GslSfResult, GslError> {
    const X_MIN: f64 = 2.0 * f64::MIN_POSITIVE;
    const X_SMALL: f64 = ROOT_EIGHT * EPSILON_SQRT; // Corrected

    let y = x.abs();

    if y == 0.0 {
        Ok(GslSfResult { val: 0.0, err: 0.0 })
    } else if y < X_MIN {
        Err(GslError::Underflow)
    } else if y < X_SMALL {
        Ok(GslSfResult { val: 0.5 * x, err: 0.0 })
    } else if y <= 3.0 {
        let c = cheb_eval(&BI1_CS, y * y / 4.5 - 1.0);
        let val = x * (0.875 + c.val);
        let mut err = y * c.err;
        err += 2.0 * f64::EPSILON * val.abs();
        Ok(GslSfResult { val, err })
    } else if y < LOG_DBL_MAX { // Corrected
        let ey = y.exp();
        // The `?` operator here elegantly handles the case where bessel_i1_scaled returns an error.
        let i1_scaled = bessel_i1_scaled(x)?;
        let val = ey * i1_scaled.val;
        let mut err = ey * i1_scaled.err + y * f64::EPSILON * val.abs();
        err += 2.0 * f64::EPSILON * val.abs();
        Ok(GslSfResult { val, err })
    } else {
        // The macro `OVERFLOW_ERROR(result)` is translated to returning an `Err`.
        Err(GslError::Overflow)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // Helper for comparing floats in tests
    fn assert_approx_eq(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() < tolerance, "{} is not close to {}", a, b);
    }

    #[test]
    fn test_i1_scaled_zero() {
        let res = bessel_i1_scaled(0.0).unwrap();
        assert_eq!(res.val, 0.0);
        assert_eq!(res.err, 0.0);
    }

    #[test]
    fn test_i1_scaled_small() {
        let res = bessel_i1_scaled(1e-9).unwrap();
        assert_eq!(res.val, 0.5e-9);
    }

    #[test]
    fn test_i1_scaled_medium() {
        // Value from WolframAlpha: N[BesselI[1, 2.5] / Exp[2.5]] = 0.224196
        let res = bessel_i1_scaled(2.5).unwrap();
        assert_approx_eq(res.val, 0.224196, 1e-6);
    }

    #[test]
    fn test_i1_scaled_large() {
        // Value from WolframAlpha: N[BesselI[1, 10.0] / Exp[10.0]] = 0.125195
        let res = bessel_i1_scaled(10.0).unwrap();
        assert_approx_eq(res.val, 0.125195, 1e-6);
    }

    #[test]
    fn test_i1_scaled_negative() {
        // I1_scaled(-x) = -I1_scaled(x)
        let res_pos = bessel_i1_scaled(5.0).unwrap();
        let res_neg = bessel_i1_scaled(-5.0).unwrap();
        assert_eq!(res_neg.val, -res_pos.val);
    }
    
    #[test]
    fn test_i1_zero() {
        let res = bessel_i1(0.0).unwrap();
        assert_eq!(res.val, 0.0);
    }

    #[test]
    fn test_i1_medium() {
        // Value from WolframAlpha: N[BesselI[1, 2.5]] = 2.73087
        let res = bessel_i1(2.5).unwrap();
        assert_approx_eq(res.val, 2.73087, 1e-5);
    }

    #[test]
    fn test_i1_overflow() {
        // LOG_DBL_MAX is around 709.78
        let res = bessel_i1(710.0);
        assert_eq!(res, Err(GslError::Overflow));
    }
}