#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use once_cell::race::OnceBox;

use crate::engine::{
    tables::{self, Multiply128lutT, Skew},
    utils, Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
};

// ======================================================================
// Avx2 - PUBLIC

/// Optimized [`Engine`] using AVX2 instructions.
///
/// [`Avx2`] is an optimized engine that follows the same algorithm as
/// [`NoSimd`] but takes advantage of the x86 AVX2 SIMD instructions.
///
/// [`NoSimd`]: crate::engine::NoSimd
#[derive(Clone, Copy)]
pub struct Avx2 {
    mul128: &'static [LutAvx2],
    skew: &'static Skew,
}

impl Avx2 {
    /// Creates new [`Avx2`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul128 = Self::get_mul128_avx2();
        let skew = tables::get_skew();

        Self { mul128, skew }
    }
}

impl Engine for Avx2 {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.fft_private_avx2(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.ifft_private_avx2(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        unsafe {
            self.mul_avx2(x, log_m);
        }
    }

    fn xor(&self, x: &mut [[u8; 64]], y: &[[u8; 64]]) {
        unsafe {
            Self::xor_avx2(x, y);
        }
    }

    fn xor_within(&self, data: &mut ShardsRefMut, x: usize, y: usize, count: usize) {
        let (xs, ys) = data.flat2_mut(x, y, count);
        unsafe {
            Self::xor_avx2(xs, ys);
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        unsafe { Self::eval_poly_avx2(erasures, truncated_size) }
    }
}

// ======================================================================
// Avx2 - IMPL Default

impl Default for Avx2 {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Avx2 - PRIVATE
//
//

#[derive(Copy, Clone)]
struct LutAvx2 {
    t0_lo: __m256i,
    t1_lo: __m256i,
    t2_lo: __m256i,
    t3_lo: __m256i,
    t0_hi: __m256i,
    t1_hi: __m256i,
    t2_hi: __m256i,
    t3_hi: __m256i,
}

impl From<&Multiply128lutT> for LutAvx2 {
    #[inline(always)]
    fn from(lut: &Multiply128lutT) -> Self {
        unsafe {
            Self {
                t0_lo: _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    core::ptr::from_ref::<u128>(&lut.lo[0]).cast::<__m128i>(),
                )),
                t1_lo: _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    core::ptr::from_ref::<u128>(&lut.lo[1]).cast::<__m128i>(),
                )),
                t2_lo: _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    core::ptr::from_ref::<u128>(&lut.lo[2]).cast::<__m128i>(),
                )),
                t3_lo: _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    core::ptr::from_ref::<u128>(&lut.lo[3]).cast::<__m128i>(),
                )),
                t0_hi: _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    core::ptr::from_ref::<u128>(&lut.hi[0]).cast::<__m128i>(),
                )),
                t1_hi: _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    core::ptr::from_ref::<u128>(&lut.hi[1]).cast::<__m128i>(),
                )),
                t2_hi: _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    core::ptr::from_ref::<u128>(&lut.hi[2]).cast::<__m128i>(),
                )),
                t3_hi: _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    core::ptr::from_ref::<u128>(&lut.hi[3]).cast::<__m128i>(),
                )),
            }
        }
    }
}

impl Avx2 {
    #[target_feature(enable = "avx2")]
    unsafe fn make_mul128_avx2() -> Vec<LutAvx2> {
        let mul128 = tables::get_mul128();
        let mut out = Vec::with_capacity(mul128.len());
        for lut in mul128.iter() {
            out.push(LutAvx2::from(lut));
        }
        out
    }

    fn get_mul128_avx2() -> &'static [LutAvx2] {
        static MUL128_AVX2: OnceBox<Vec<LutAvx2>> = OnceBox::new();
        let table = MUL128_AVX2.get_or_init(|| Box::new(unsafe { Self::make_mul128_avx2() }));
        table.as_slice()
    }

    #[target_feature(enable = "avx2")]
    unsafe fn xor_avx2(x: &mut [[u8; 64]], y: &[[u8; 64]]) {
        debug_assert_eq!(x.len(), y.len());

        let len = x.len();
        let mut x_chunk = x.as_mut_ptr();
        let mut y_chunk = y.as_ptr();
        for _ in 0..len {
            unsafe {
                let x_ptr = (*x_chunk).as_mut_ptr().cast::<__m256i>();
                let y_ptr = (*y_chunk).as_ptr().cast::<__m256i>();
                let x_lo = _mm256_loadu_si256(x_ptr);
                let x_hi = _mm256_loadu_si256(x_ptr.add(1));
                let y_lo = _mm256_loadu_si256(y_ptr);
                let y_hi = _mm256_loadu_si256(y_ptr.add(1));

                _mm256_storeu_si256(x_ptr, _mm256_xor_si256(x_lo, y_lo));
                _mm256_storeu_si256(x_ptr.add(1), _mm256_xor_si256(x_hi, y_hi));

                x_chunk = x_chunk.add(1);
                y_chunk = y_chunk.add(1);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut_avx2 = &self.mul128[log_m as usize];

        let len = x.len();
        let mut chunk = x.as_mut_ptr();
        for _ in 0..len {
            unsafe {
                let x_ptr = (*chunk).as_mut_ptr().cast::<__m256i>();
                let x_lo = _mm256_loadu_si256(x_ptr);
                let x_hi = _mm256_loadu_si256(x_ptr.add(1));
                let (prod_lo, prod_hi) = Self::mul_256(x_lo, x_hi, lut_avx2);
                _mm256_storeu_si256(x_ptr, prod_lo);
                _mm256_storeu_si256(x_ptr.add(1), prod_hi);

                chunk = chunk.add(1);
            }
        }
    }

    // Impelemntation of LEO_MUL_256
    #[inline(always)]
    fn mul_256(value_lo: __m256i, value_hi: __m256i, lut_avx2: &LutAvx2) -> (__m256i, __m256i) {
        let mut prod_lo: __m256i;
        let mut prod_hi: __m256i;

        unsafe {
            let clr_mask = _mm256_set1_epi8(0x0f);

            let data_0 = _mm256_and_si256(value_lo, clr_mask);
            prod_lo = _mm256_shuffle_epi8(lut_avx2.t0_lo, data_0);
            prod_hi = _mm256_shuffle_epi8(lut_avx2.t0_hi, data_0);

            let data_1 = _mm256_and_si256(_mm256_srli_epi64(value_lo, 4), clr_mask);
            prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(lut_avx2.t1_lo, data_1));
            prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(lut_avx2.t1_hi, data_1));

            let data_0 = _mm256_and_si256(value_hi, clr_mask);
            prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(lut_avx2.t2_lo, data_0));
            prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(lut_avx2.t2_hi, data_0));

            let data_1 = _mm256_and_si256(_mm256_srli_epi64(value_hi, 4), clr_mask);
            prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(lut_avx2.t3_lo, data_1));
            prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(lut_avx2.t3_hi, data_1));
        }

        (prod_lo, prod_hi)
    }

    //// {x_lo, x_hi} ^= {y_lo, y_hi} * log_m
    // Implementation of LEO_MULADD_256
    #[inline(always)]
    fn muladd_256(
        mut x_lo: __m256i,
        mut x_hi: __m256i,
        y_lo: __m256i,
        y_hi: __m256i,
        lut_avx2: &LutAvx2,
    ) -> (__m256i, __m256i) {
        let (prod_lo, prod_hi) = Self::mul_256(y_lo, y_hi, lut_avx2);
        unsafe {
            x_lo = _mm256_xor_si256(x_lo, prod_lo);
            x_hi = _mm256_xor_si256(x_hi, prod_hi);
        }
        (x_lo, x_hi)
    }

    // Implementation of LEO_FFTB_256 for register operands.
    #[inline(always)]
    fn fftb_256_reg<const MUL: bool>(
        mut x_lo: __m256i,
        mut x_hi: __m256i,
        mut y_lo: __m256i,
        mut y_hi: __m256i,
        lut_avx2: &LutAvx2,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        if MUL {
            (x_lo, x_hi) = Self::muladd_256(x_lo, x_hi, y_lo, y_hi, lut_avx2);
        }
        unsafe {
            y_lo = _mm256_xor_si256(y_lo, x_lo);
            y_hi = _mm256_xor_si256(y_hi, x_hi);
        }
        (x_lo, x_hi, y_lo, y_hi)
    }

    // Implementation of LEO_IFFTB_256 for register operands.
    #[inline(always)]
    fn ifftb_256_reg<const MUL: bool>(
        mut x_lo: __m256i,
        mut x_hi: __m256i,
        mut y_lo: __m256i,
        mut y_hi: __m256i,
        lut_avx2: &LutAvx2,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            y_lo = _mm256_xor_si256(y_lo, x_lo);
            y_hi = _mm256_xor_si256(y_hi, x_hi);
        }
        if MUL {
            (x_lo, x_hi) = Self::muladd_256(x_lo, x_hi, y_lo, y_hi, lut_avx2);
        }
        (x_lo, x_hi, y_lo, y_hi)
    }
}

// ======================================================================
// Avx2 - PRIVATE - FFT (fast Fourier transform)

impl Avx2 {
    // Implementation of LEO_FFTB_256
    #[inline(always)]
    fn fftb_256(x: &mut [u8; 64], y: &mut [u8; 64], lut_avx2: &LutAvx2) {
        let x_ptr = x.as_mut_ptr().cast::<__m256i>();
        let y_ptr = y.as_mut_ptr().cast::<__m256i>();

        unsafe {
            let x_lo = _mm256_loadu_si256(x_ptr);
            let x_hi = _mm256_loadu_si256(x_ptr.add(1));
            let y_lo = _mm256_loadu_si256(y_ptr);
            let y_hi = _mm256_loadu_si256(y_ptr.add(1));
            let (x_lo, x_hi, y_lo, y_hi) =
                Self::fftb_256_reg::<true>(x_lo, x_hi, y_lo, y_hi, lut_avx2);
            _mm256_storeu_si256(x_ptr, x_lo);
            _mm256_storeu_si256(x_ptr.add(1), x_hi);
            _mm256_storeu_si256(y_ptr, y_lo);
            _mm256_storeu_si256(y_ptr.add(1), y_hi);
        }
    }

    #[inline(always)]
    fn fft_butterfly_partial_lut(x: &mut [[u8; 64]], y: &mut [[u8; 64]], lut_avx2: &LutAvx2) {
        debug_assert_eq!(x.len(), y.len());
        let len = x.len();
        let mut x_chunk = x.as_mut_ptr();
        let mut y_chunk = y.as_mut_ptr();

        for _ in 0..len {
            unsafe {
                Self::fftb_256(&mut *x_chunk, &mut *y_chunk, lut_avx2);
                x_chunk = x_chunk.add(1);
                y_chunk = y_chunk.add(1);
            }
        }
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut_avx2 = &self.mul128[log_m as usize];
        Self::fft_butterfly_partial_lut(x, y, lut_avx2);
    }

    #[inline(always)]
    fn fft_butterfly_two_layers_lut<
        const MUL_M01: bool,
        const MUL_M23: bool,
        const MUL_M02: bool,
    >(
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        lut_m01: &LutAvx2,
        lut_m23: &LutAvx2,
        lut_m02: &LutAvx2,
    ) {
        let (s0, s1, s2, s3) = data.dist4_flat_mut(pos, dist);
        debug_assert_eq!(s0.len(), s1.len());
        debug_assert_eq!(s0.len(), s2.len());
        debug_assert_eq!(s0.len(), s3.len());

        // Fuse two FFT layers per chunk to reduce memory traffic.
        let len = s0.len();
        let mut s0_ptr = s0.as_mut_ptr();
        let mut s1_ptr = s1.as_mut_ptr();
        let mut s2_ptr = s2.as_mut_ptr();
        let mut s3_ptr = s3.as_mut_ptr();

        for _ in 0..len {
            unsafe {
                let p0 = (*s0_ptr).as_mut_ptr().cast::<__m256i>();
                let p1 = (*s1_ptr).as_mut_ptr().cast::<__m256i>();
                let p2 = (*s2_ptr).as_mut_ptr().cast::<__m256i>();
                let p3 = (*s3_ptr).as_mut_ptr().cast::<__m256i>();

                let mut s0_lo = _mm256_loadu_si256(p0);
                let mut s0_hi = _mm256_loadu_si256(p0.add(1));
                let mut s1_lo = _mm256_loadu_si256(p1);
                let mut s1_hi = _mm256_loadu_si256(p1.add(1));
                let mut s2_lo = _mm256_loadu_si256(p2);
                let mut s2_hi = _mm256_loadu_si256(p2.add(1));
                let mut s3_lo = _mm256_loadu_si256(p3);
                let mut s3_hi = _mm256_loadu_si256(p3.add(1));

                (s0_lo, s0_hi, s2_lo, s2_hi) =
                    Self::fftb_256_reg::<MUL_M02>(s0_lo, s0_hi, s2_lo, s2_hi, lut_m02);
                (s1_lo, s1_hi, s3_lo, s3_hi) =
                    Self::fftb_256_reg::<MUL_M02>(s1_lo, s1_hi, s3_lo, s3_hi, lut_m02);

                (s0_lo, s0_hi, s1_lo, s1_hi) =
                    Self::fftb_256_reg::<MUL_M01>(s0_lo, s0_hi, s1_lo, s1_hi, lut_m01);
                (s2_lo, s2_hi, s3_lo, s3_hi) =
                    Self::fftb_256_reg::<MUL_M23>(s2_lo, s2_hi, s3_lo, s3_hi, lut_m23);

                _mm256_storeu_si256(p0, s0_lo);
                _mm256_storeu_si256(p0.add(1), s0_hi);
                _mm256_storeu_si256(p1, s1_lo);
                _mm256_storeu_si256(p1.add(1), s1_hi);
                _mm256_storeu_si256(p2, s2_lo);
                _mm256_storeu_si256(p2.add(1), s2_hi);
                _mm256_storeu_si256(p3, s3_lo);
                _mm256_storeu_si256(p3.add(1), s3_hi);

                s0_ptr = s0_ptr.add(1);
                s1_ptr = s1_ptr.add(1);
                s2_ptr = s2_ptr.add(1);
                s3_ptr = s3_ptr.add(1);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn fft_private_avx2(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.fft_private(data, pos, size, truncated_size, skew_delta);
    }

    #[inline(always)]
    fn fft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist4 = size;
        let mut dist = size >> 2;
        while dist != 0 {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                let lut_m01 = (log_m01 != GF_MODULUS).then_some(&self.mul128[log_m01 as usize]);
                let lut_m02 = (log_m02 != GF_MODULUS).then_some(&self.mul128[log_m02 as usize]);
                let lut_m23 = (log_m23 != GF_MODULUS).then_some(&self.mul128[log_m23 as usize]);
                let lut_id = &self.mul128[0];

                match (lut_m01, lut_m23, lut_m02) {
                    (Some(m01), Some(m23), Some(m02)) => Self::fft_butterfly_two_layers_lut::<
                        true,
                        true,
                        true,
                    >(
                        data, pos + r, dist, m01, m23, m02
                    ),
                    (Some(m01), Some(m23), None) => Self::fft_butterfly_two_layers_lut::<
                        true,
                        true,
                        false,
                    >(
                        data, pos + r, dist, m01, m23, lut_id
                    ),
                    (Some(m01), None, Some(m02)) => Self::fft_butterfly_two_layers_lut::<
                        true,
                        false,
                        true,
                    >(
                        data, pos + r, dist, m01, lut_id, m02
                    ),
                    (Some(m01), None, None) => Self::fft_butterfly_two_layers_lut::<
                        true,
                        false,
                        false,
                    >(
                        data, pos + r, dist, m01, lut_id, lut_id
                    ),
                    (None, Some(m23), Some(m02)) => Self::fft_butterfly_two_layers_lut::<
                        false,
                        true,
                        true,
                    >(
                        data, pos + r, dist, lut_id, m23, m02
                    ),
                    (None, Some(m23), None) => Self::fft_butterfly_two_layers_lut::<
                        false,
                        true,
                        false,
                    >(
                        data, pos + r, dist, lut_id, m23, lut_id
                    ),
                    (None, None, Some(m02)) => Self::fft_butterfly_two_layers_lut::<
                        false,
                        false,
                        true,
                    >(
                        data, pos + r, dist, lut_id, lut_id, m02
                    ),
                    (None, None, None) => {
                        Self::fft_butterfly_two_layers_lut::<false, false, false>(
                            data,
                            pos + r,
                            dist,
                            lut_id,
                            lut_id,
                            lut_id,
                        )
                    }
                }

                r += dist4;
            }
            dist4 = dist;
            dist >>= 2;
        }

        // FINAL ODD LAYER

        if dist4 == 2 {
            let mut r = 0;
            while r < truncated_size {
                let log_m = self.skew[r + skew_delta];

                let (x, y) = data.dist2_mut(pos + r, 1);

                if log_m == GF_MODULUS {
                    self.xor(y, x);
                } else {
                    self.fft_butterfly_partial(x, y, log_m);
                }

                r += 2;
            }
        }
    }
}

// ======================================================================
// Avx2 - PRIVATE - IFFT (inverse fast Fourier transform)

impl Avx2 {
    // Implementation of LEO_IFFTB_256
    #[inline(always)]
    fn ifftb_256(x: &mut [u8; 64], y: &mut [u8; 64], lut_avx2: &LutAvx2) {
        let x_ptr = x.as_mut_ptr().cast::<__m256i>();
        let y_ptr = y.as_mut_ptr().cast::<__m256i>();

        unsafe {
            let x_lo = _mm256_loadu_si256(x_ptr);
            let x_hi = _mm256_loadu_si256(x_ptr.add(1));
            let y_lo = _mm256_loadu_si256(y_ptr);
            let y_hi = _mm256_loadu_si256(y_ptr.add(1));
            let (x_lo, x_hi, y_lo, y_hi) =
                Self::ifftb_256_reg::<true>(x_lo, x_hi, y_lo, y_hi, lut_avx2);
            _mm256_storeu_si256(x_ptr, x_lo);
            _mm256_storeu_si256(x_ptr.add(1), x_hi);
            _mm256_storeu_si256(y_ptr, y_lo);
            _mm256_storeu_si256(y_ptr.add(1), y_hi);
        }
    }

    #[inline(always)]
    fn ifft_butterfly_partial_lut(x: &mut [[u8; 64]], y: &mut [[u8; 64]], lut_avx2: &LutAvx2) {
        debug_assert_eq!(x.len(), y.len());
        let len = x.len();
        let mut x_chunk = x.as_mut_ptr();
        let mut y_chunk = y.as_mut_ptr();

        for _ in 0..len {
            unsafe {
                Self::ifftb_256(&mut *x_chunk, &mut *y_chunk, lut_avx2);
                x_chunk = x_chunk.add(1);
                y_chunk = y_chunk.add(1);
            }
        }
    }

    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut_avx2 = &self.mul128[log_m as usize];
        Self::ifft_butterfly_partial_lut(x, y, lut_avx2);
    }

    #[inline(always)]
    fn ifft_butterfly_two_layers_lut<
        const MUL_M01: bool,
        const MUL_M23: bool,
        const MUL_M02: bool,
    >(
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        lut_m01: &LutAvx2,
        lut_m23: &LutAvx2,
        lut_m02: &LutAvx2,
    ) {
        let (s0, s1, s2, s3) = data.dist4_flat_mut(pos, dist);
        debug_assert_eq!(s0.len(), s1.len());
        debug_assert_eq!(s0.len(), s2.len());
        debug_assert_eq!(s0.len(), s3.len());

        // Fuse two IFFT layers per chunk to reduce memory traffic.
        let len = s0.len();
        let mut s0_ptr = s0.as_mut_ptr();
        let mut s1_ptr = s1.as_mut_ptr();
        let mut s2_ptr = s2.as_mut_ptr();
        let mut s3_ptr = s3.as_mut_ptr();

        for _ in 0..len {
            unsafe {
                let p0 = (*s0_ptr).as_mut_ptr().cast::<__m256i>();
                let p1 = (*s1_ptr).as_mut_ptr().cast::<__m256i>();
                let p2 = (*s2_ptr).as_mut_ptr().cast::<__m256i>();
                let p3 = (*s3_ptr).as_mut_ptr().cast::<__m256i>();

                let mut s0_lo = _mm256_loadu_si256(p0);
                let mut s0_hi = _mm256_loadu_si256(p0.add(1));
                let mut s1_lo = _mm256_loadu_si256(p1);
                let mut s1_hi = _mm256_loadu_si256(p1.add(1));
                let mut s2_lo = _mm256_loadu_si256(p2);
                let mut s2_hi = _mm256_loadu_si256(p2.add(1));
                let mut s3_lo = _mm256_loadu_si256(p3);
                let mut s3_hi = _mm256_loadu_si256(p3.add(1));

                (s0_lo, s0_hi, s1_lo, s1_hi) =
                    Self::ifftb_256_reg::<MUL_M01>(s0_lo, s0_hi, s1_lo, s1_hi, lut_m01);
                (s2_lo, s2_hi, s3_lo, s3_hi) =
                    Self::ifftb_256_reg::<MUL_M23>(s2_lo, s2_hi, s3_lo, s3_hi, lut_m23);

                (s0_lo, s0_hi, s2_lo, s2_hi) =
                    Self::ifftb_256_reg::<MUL_M02>(s0_lo, s0_hi, s2_lo, s2_hi, lut_m02);
                (s1_lo, s1_hi, s3_lo, s3_hi) =
                    Self::ifftb_256_reg::<MUL_M02>(s1_lo, s1_hi, s3_lo, s3_hi, lut_m02);

                _mm256_storeu_si256(p0, s0_lo);
                _mm256_storeu_si256(p0.add(1), s0_hi);
                _mm256_storeu_si256(p1, s1_lo);
                _mm256_storeu_si256(p1.add(1), s1_hi);
                _mm256_storeu_si256(p2, s2_lo);
                _mm256_storeu_si256(p2.add(1), s2_hi);
                _mm256_storeu_si256(p3, s3_lo);
                _mm256_storeu_si256(p3.add(1), s3_hi);

                s0_ptr = s0_ptr.add(1);
                s1_ptr = s1_ptr.add(1);
                s2_ptr = s2_ptr.add(1);
                s3_ptr = s3_ptr.add(1);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn ifft_private_avx2(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.ifft_private(data, pos, size, truncated_size, skew_delta);
    }

    #[inline(always)]
    fn ifft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist = 1;
        let mut dist4 = 4;
        while dist4 <= size {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                let lut_m01 = (log_m01 != GF_MODULUS).then_some(&self.mul128[log_m01 as usize]);
                let lut_m02 = (log_m02 != GF_MODULUS).then_some(&self.mul128[log_m02 as usize]);
                let lut_m23 = (log_m23 != GF_MODULUS).then_some(&self.mul128[log_m23 as usize]);
                let lut_id = &self.mul128[0];

                match (lut_m01, lut_m23, lut_m02) {
                    (Some(m01), Some(m23), Some(m02)) => Self::ifft_butterfly_two_layers_lut::<
                        true,
                        true,
                        true,
                    >(
                        data, pos + r, dist, m01, m23, m02
                    ),
                    (Some(m01), Some(m23), None) => Self::ifft_butterfly_two_layers_lut::<
                        true,
                        true,
                        false,
                    >(
                        data, pos + r, dist, m01, m23, lut_id
                    ),
                    (Some(m01), None, Some(m02)) => Self::ifft_butterfly_two_layers_lut::<
                        true,
                        false,
                        true,
                    >(
                        data, pos + r, dist, m01, lut_id, m02
                    ),
                    (Some(m01), None, None) => Self::ifft_butterfly_two_layers_lut::<
                        true,
                        false,
                        false,
                    >(
                        data, pos + r, dist, m01, lut_id, lut_id
                    ),
                    (None, Some(m23), Some(m02)) => Self::ifft_butterfly_two_layers_lut::<
                        false,
                        true,
                        true,
                    >(
                        data, pos + r, dist, lut_id, m23, m02
                    ),
                    (None, Some(m23), None) => Self::ifft_butterfly_two_layers_lut::<
                        false,
                        true,
                        false,
                    >(
                        data, pos + r, dist, lut_id, m23, lut_id
                    ),
                    (None, None, Some(m02)) => Self::ifft_butterfly_two_layers_lut::<
                        false,
                        false,
                        true,
                    >(
                        data, pos + r, dist, lut_id, lut_id, m02
                    ),
                    (None, None, None) => {
                        Self::ifft_butterfly_two_layers_lut::<false, false, false>(
                            data,
                            pos + r,
                            dist,
                            lut_id,
                            lut_id,
                            lut_id,
                        )
                    }
                }

                r += dist4;
            }
            dist = dist4;
            dist4 <<= 2;
        }

        // FINAL ODD LAYER

        if dist < size {
            let log_m = self.skew[dist + skew_delta - 1];
            if log_m == GF_MODULUS {
                self.xor_within(data, pos + dist, pos, dist);
            } else {
                let (mut a, mut b) = data.split_at_mut(pos + dist);
                for i in 0..dist {
                    self.ifft_butterfly_partial(
                        &mut a[pos + i], // data[pos + i]
                        &mut b[i],       // data[pos + i + dist]
                        log_m,
                    );
                }
            }
        }
    }
}

// ======================================================================
// Avx2 - PRIVATE - Evaluate polynomial

impl Avx2 {
    #[target_feature(enable = "avx2")]
    unsafe fn eval_poly_avx2(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        utils::eval_poly(erasures, truncated_size);
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
