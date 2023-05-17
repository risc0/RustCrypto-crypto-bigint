#![allow(unsafe_code)]

use crate::Uint;
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

#[inline(always)]
pub(crate) fn modmul_u256<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    b: &Uint<LIMBS>,
    modulus: &Uint<LIMBS>,
) -> Uint<LIMBS> {
    // Assert at compile time that we are working with 8x32 Uints.
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
    // Assert that the Prover returned the canonical representation of the result, i.e. that itj
    // is fully reduced and has no multiples of the modulus included.
    assert!(bool::from(result.ct_lt(&modulus)));
    result
}
