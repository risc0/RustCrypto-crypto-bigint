use crate::{Limb, Uint};

use super::reduction::montgomery_reduction;

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use crate::risc0;

pub(crate) fn into_montgomery_form<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    r2: &Uint<LIMBS>,
    modulus: &Uint<LIMBS>,
    mod_neg_inv: Limb,
) -> Uint<LIMBS> {
    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    if LIMBS == risc0::BIGINT_WIDTH_WORDS {
        // Ensure that the input is reduced by passing it though a modmul by one.
        return risc0::modmul_u256(a, &Uint::<LIMBS>::ONE, modulus);
    }

    let product = a.mul_wide(r2);
    montgomery_reduction::<LIMBS>(&product, modulus, mod_neg_inv)
}

pub(crate) fn from_montgomery_form<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    modulus: &Uint<LIMBS>,
    mod_neg_inv: Limb,
) -> Uint<LIMBS> {
    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    if LIMBS == risc0::BIGINT_WIDTH_WORDS {
        // In the RISC Zero zkVM 256-bit residues are represented in standard form instead of
        // Montgomery because, with the accelerator, multiplication is more efficient.
        return a.clone();
    }

    montgomery_reduction::<LIMBS>(&(*a, Uint::<LIMBS>::ZERO), modulus, mod_neg_inv)
}
