use std::f64;

// --- Coefficients for Chebyshev Approximations ---
// These are translated from the C code's #ifdef UNK block.
// In Rust, we declare them as static arrays of f64.

// Chebyshev coefficients for exp(-x) I1(x) / x
// in the interval [0,8]. (n = 29 coefficients)
static A_COEFFS: [f64; 29] = [
    2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
    1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
    7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
    3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
    1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
    3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
    9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
    2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
    3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
    4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
    5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
    4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
    2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
    1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
    2.52587186443633654823E-1,
];

// Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
// in the inverted interval [8,infinity]. (n = 25 coefficients)
static B_COEFFS: [f64; 25] = [
    7.51729631084210481353E-18,
    4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
    2.96262899764595013876E-16,
    3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
    1.04202769841288027642E-14,
    4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
    2.03562854414708950722E-12,
    1.41258074366137813316E-11,
    3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
    7.78576235018280120474E-1,
];

// --- Chebyshev Series Evaluation (Translated from chbevl.c) ---

/// Evaluates a Chebyshev series at a given point `x`.
///
/// This function implements Clenshaw's recurrence formula.
///
/// # Arguments
/// * `x`: The argument at which to evaluate the series.
/// * `coeffs`: A slice of `f64` representing the Chebyshev coefficients,
///            ordered from `c_0` to `c_{n-1}`.
#[inline(always)]
fn chbevl<const N: usize>(x: f64, coeffs: &[f64; N]) -> f64 {
    // N is compile-time constant (25 or 29). Keep a debug assert for safety.
    debug_assert!(N >= 2);

    // Initialize b0 with the first coefficient (array[0])
    let mut b0 = coeffs[0];
    let mut b1 = 0.0f64;
    let mut b2 = 0.0f64;

    // The C code's `do-while(--i)` loop effectively runs `n-1` times,
    // processing coefficients from `array[1]` to `array[n-1]`.
    // A `for` loop is safer and more idiomatic in Rust.
    for i in 1..N {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + coeffs[i];
    }

    0.5 * (b0 - b2)
}

// --- Exponentially Scaled Modified Bessel Function (Translated from i1e.c) ---

/// Calculates the exponentially scaled modified Bessel function of the first kind
/// of order one, I_1(x) * exp(-|x|).
///
/// This version is useful for large `x` where I_1(x) itself might overflow.
/// The function uses Chebyshev polynomial expansions for two intervals:
/// [0, 8] and (8, infinity).
///
/// # Arguments
/// * `x`: The argument for which to calculate I_1(x) * exp(-|x|).
///
/// # Returns
/// The value of I_1(x) * exp(-|x|).
pub fn i1e(x: f64) -> f64 {
    let z = x.abs();

    let result = if z <= 8.0 {
    
        let y = (z * 0.5) - 2.0;
        chbevl(y, &A_COEFFS) * z
    } else {
        
        let z_inv = 1.0 / z;
        let sqrt_z_inv = (z_inv).sqrt();
        chbevl(32.0 * z_inv - 2.0, &B_COEFFS) * sqrt_z_inv
    };

    result.copysign(x)
}

// --- Modified Bessel Function of Order One (Translated from i1.c) ---

/// Calculates the modified Bessel function of the first kind of order one, I_1(x).
///
/// The function uses Chebyshev polynomial expansions for two intervals:
/// [0, 8] and (8, infinity).
///
/// # Arguments
/// * `x`: The argument for which to calculate I_1(x).
///
/// # Returns
/// The value of I_1(x).
pub fn i1(x: f64) -> f64 {
    let z = x.abs(); 

    let result = if z <= 8.0 {

        let y = (z * 0.5) - 2.0;
        chbevl(y, &A_COEFFS) * z * z.exp()
    } else {
        let z_inv = 1.0 / z;
        let sqrt_z_inv = (z_inv).sqrt();
        z.exp() * chbevl(32.0 *z_inv - 2.0, &B_COEFFS) * sqrt_z_inv
    };

    result.copysign(x)
}