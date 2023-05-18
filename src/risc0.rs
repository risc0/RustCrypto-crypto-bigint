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

#[inline(always)]
pub(crate) fn mul_wide_u128<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    b: &Uint<LIMBS>,
) -> (Uint<LIMBS>, Uint<LIMBS>) {
    // Assert at compile time that we are working with 4x32 Uints.
    assert!(LIMBS == BIGINT_WIDTH_WORDS / 2);

    let mut a_pad = [0u32; BIGINT_WIDTH_WORDS];
    a_pad[..LIMBS].copy_from_slice(a.as_words());
    let mut b_pad = [0u32; BIGINT_WIDTH_WORDS];
    b_pad[..LIMBS].copy_from_slice(b.as_words());

    let result = unsafe {
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
    };
    let (lo, hi) = result.split_at(LIMBS);

    (
        Uint::<{ LIMBS }>::from_words(lo.try_into().unwrap()),
        Uint::<{ LIMBS }>::from_words(hi.try_into().unwrap()),
    )
}
