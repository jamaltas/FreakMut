use num_traits::{Num, Float, FromPrimitive}; // You'll need the `num-traits` crate for these

// Your provided InterpMode definition (make sure it matches exactly what you have)
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum InterpMode<T> {
    Extrapolate,
    FirstLast,
    Constant(T),
}

// Your provided prev_index function (already zero-allocation)
#[inline]
fn prev_index<T>(x: &[T], xp: T) -> usize
where
    T: Num + PartialOrd + Copy,
{
    // `partition_point` returns the index of the first element `e` such that `e >= xp`.
    // We want the index `i` such that `x[i] <= xp`, so we subtract 1.
    // `saturating_sub(1)` handles cases where `xp < x[0]` by clamping to 0.
    x.partition_point(|&probe| probe < xp).saturating_sub(1)
}

// Your provided select_outside_point function (already zero-allocation)
fn select_outside_point<T>(
    x_limits: (&T, &T),
    y_limits: (&T, &T),
    xp: &T,
    default: T, // This is the interpolated value IF xp is within the x_limits
    mode: &InterpMode<T>,
) -> T
where
    T: Num + PartialOrd + Copy,
{
    if xp < x_limits.0 {
        match mode {
            InterpMode::Extrapolate => default,
            InterpMode::FirstLast => *y_limits.0,
            InterpMode::Constant(val) => *val,
        }
    } else if xp > x_limits.1 {
        match mode {
            InterpMode::Extrapolate => default,
            InterpMode::FirstLast => *y_limits.1,
            InterpMode::Constant(val) => *val,
        }
    } else {
        default // xp is within x_limits, so return the linearly interpolated value
    }
}


/// Performs linear interpolation without any internal memory allocations.
///
/// This function calculates the slope and intercept for the relevant segment
/// on the fly, avoiding the creation of intermediate vectors for deltas,
/// slopes, and intercepts.
///
/// Assumes `x` is monotonically increasing. Behavior for non-monotonic `x`
/// is undefined, as it relies on `partition_point`.
///
/// # Type Parameters
/// `T`: Must be a floating-point type (e.g., `f32`, `f64`) to handle division by zero
///      (resulting in `inf` or `NaN`) and to provide `is_nan()` and `nan()` methods.
pub fn interp_zero_alloc<T>(x: &[T], y: &[T], xp: T, mode: &InterpMode<T>) -> T
where
    T: Float + PartialOrd + Copy + FromPrimitive + std::fmt::Debug,
{
    let min_len = std::cmp::min(x.len(), y.len());

    // Handle edge cases for empty or single-point data
    if min_len == 0 {
        return T::zero(); // Or T::nan() if an undefined result is preferred for empty input
    } else if min_len == 1 {
        return y[0];
    }

    // Find the index `i` of the segment's starting point `(x[i], y[i])`.
    // The segment for interpolation will be from `(x[i], y[i])` to `(x[i+1], y[i+1])`.
    // `.min(min_len - 2)` ensures that `i` (and thus `i+1`) are always valid indices
    // for accessing `x` and `y` slices.
    let i = prev_index(x, xp).min(min_len - 2);

    let x0 = x[i];
    let y0 = y[i];
    let x1 = x[i + 1];
    let y1 = y[i + 1];

    let point_interp;

    // Calculate delta_x for the current segment
    let dx_local = x1 - x0;

    // Handle vertical segments (where x0 == x1)
    if dx_local == T::zero() {
        if xp == x0 {
            // If xp exactly matches the x-coordinate of the vertical segment,
            // return the corresponding y-value (y0 or y1). This is a common
            // interpretation for exact matches on vertical lines.
            point_interp = y0;
        } else {
            // If xp is not on this vertical segment, linear interpolation is undefined.
            // Return NaN to indicate an invalid result.
            point_interp = T::nan();
        }
    } else {
        // Normal segment: calculate slope and linearly interpolate
        let m_i = (y1 - y0) / dx_local;
        // The interpolation formula: y = y0 + m * (xp - x0)
        point_interp = y0 + m_i * (xp - x0);
    }

    // After calculating the linearly interpolated value (`point_interp`),
    // apply the `InterpMode` logic if `xp` falls outside the range [x[0], x[min_len-1]].
    let x_limits = (&x[0], &x[min_len - 1]);
    let y_limits = (&y[0], &y[min_len - 1]); // These are used by `FirstLast` mode

    select_outside_point(x_limits, y_limits, &xp, point_interp, mode)
}