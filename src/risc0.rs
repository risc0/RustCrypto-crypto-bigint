#![allow(unsafe_code)]

use crate::{Uint, U128, U256};
use subtle::ConstantTimeLess;

/// RISC Zero supports BigInt operations with a width of 256-bits as 8x32-bit words.
pub(crate) const BIGINT_WIDTH_WORDS: usize = 8;
const OP_MULTIPLY: u32 = 0;

extern "C" {
    fn sys_bigint(
        result: *mut [u32; BIGINT_WIDTH_WORDS],
        op: u32,
        x: *const [u32; BIGINT_WIDTH_WORDS],
        y: *const [u32; BIGINT_WIDTH_WORDS],
        modulus: *const [u32; BIGINT_WIDTH_WORDS],
    );
}

/// Modular multiplication of two 256-bit Uint values using the RISC Zero accelerator.
/// Returns the fully reduced and normalized modular multiplication result.
///
/// NOTE: This method takes generic Uint values, but asserts that the input is 256 bits. It is
/// provided because the main places we want to patch in multiplication use the generic Uint type,
/// and specialization is not a stable Rust feature. When inlined, the assert should be removed.
#[inline(always)]
pub(crate) fn modmul_uint_256<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    b: &Uint<LIMBS>,
    modulus: &Uint<LIMBS>,
) -> Uint<LIMBS> {
    // Assert that we are working with 8x32 Uints.
    assert!(LIMBS == BIGINT_WIDTH_WORDS);

    let result = Uint::<LIMBS>::from_words(unsafe {
        let mut out = core::mem::MaybeUninit::<[u32; LIMBS]>::uninit();
        sys_bigint(
            out.as_mut_ptr() as *mut [u32; BIGINT_WIDTH_WORDS],
            OP_MULTIPLY,
            a.as_words().as_ptr() as *const [u32; BIGINT_WIDTH_WORDS],
            b.as_words().as_ptr() as *const [u32; BIGINT_WIDTH_WORDS],
            modulus.as_words().as_ptr() as *const [u32; BIGINT_WIDTH_WORDS],
        );
        out.assume_init()
    });
    // Assert that the Prover returned the canonical representation of the result, i.e. that it
    // is fully reduced and has no multiples of the modulus included.
    // NOTE: On a cooperating prover, this check will always evaluate to false, and therefore
    // will have timing invariant with any secrets. If the prover is faulty, this check may
    // leak secret information through timing, however this is out of scope since a faulty
    // cannot be relied upon for the privacy of the inputs.
    assert!(bool::from(result.ct_lt(&modulus)));
    result
}

/// Wide multiplication of two 128-bit Uint values using the RISC Zero accelerator.
#[inline(always)]
pub fn mul_wide_u128(a: &U128, b: &U128) -> U256 {
    let mut a_pad = [0u32; BIGINT_WIDTH_WORDS];
    a_pad[..U128::LIMBS].copy_from_slice(a.as_words());
    let mut b_pad = [0u32; BIGINT_WIDTH_WORDS];
    b_pad[..U128::LIMBS].copy_from_slice(b.as_words());

    U256::from_words(unsafe {
        let mut out = core::mem::MaybeUninit::<[u32; BIGINT_WIDTH_WORDS]>::uninit();
        // sys_bigint with modulus set to use is a wide u128 multiplication.
        sys_bigint(
            out.as_mut_ptr(),
            OP_MULTIPLY,
            a_pad.as_ptr() as *const [u32; BIGINT_WIDTH_WORDS],
            b_pad.as_ptr() as *const [u32; BIGINT_WIDTH_WORDS],
            &[0u32; BIGINT_WIDTH_WORDS],
        );
        out.assume_init()
    })
}

/// Modular multiplication of two 256-bit Uint values using the RISC Zero accelerator.
/// Returns a result in the equivelence class of integers under the modulus, but the result is not
/// guarenteed to be fully reduced. In particular, it may include any number of multiples of the
/// modulus.
#[inline(always)]
pub fn modmul_u256_denormalized(a: &U256, b: &U256, modulus: &U256) -> U256 {
    U256::from_words(unsafe {
        let mut out = core::mem::MaybeUninit::<[u32; BIGINT_WIDTH_WORDS]>::uninit();
        sys_bigint(
            out.as_mut_ptr(),
            OP_MULTIPLY,
            a.as_words(),
            b.as_words(),
            modulus.as_words(),
        );
        out.assume_init()
    })
}

/// Modular multiplication of two 256-bit Uint values using the RISC Zero accelerator.
/// Returns the fully reduced and normalized modular multiplication result.
#[inline(always)]
pub fn modmul_u256(a: &U256, b: &U256, modulus: &U256) -> U256 {
    let result = modmul_u256_denormalized(a, b, modulus);
    // Assert that the Prover returned the canonical representation of the result, i.e. that it
    // is fully reduced and has no multiples of the modulus included.
    // NOTE: On a cooperating prover, this check will always evaluate to false, and therefore
    // will have timing invariant with any secrets. If the prover is faulty, this check may
    // leak secret information through timing, however this is out of scope since a faulty
    // cannot be relied upon for the privacy of the inputs.
    assert!(bool::from(result.ct_lt(&modulus)));
    result
}
