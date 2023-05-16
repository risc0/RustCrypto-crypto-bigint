use crate::{Limb, Uint};

use super::reduction::montgomery_reduction;

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use risc0_zkvm_platform::syscall::{bigint, sys_bigint};

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use subtle::ConstantTimeLess;

pub(crate) fn into_montgomery_form<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    r2: &Uint<LIMBS>,
    modulus: &Uint<LIMBS>,
    mod_neg_inv: Limb,
    _r: &Uint<LIMBS>,
) -> Uint<LIMBS> {
    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    if LIMBS == bigint::WIDTH_WORDS {
        // In the RISC Zero zkVM 256-bit residues are represented in standard form instead of
        // Montgomery because, with the accelerator, multiplication is more efficient.
        return a.clone();
    }

    let product = a.mul_wide(r2);
    montgomery_reduction::<LIMBS>(&product, modulus, mod_neg_inv)
}

pub(crate) fn from_montgomery_form<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    modulus: &Uint<LIMBS>,
    mod_neg_inv: Limb,
    _r_inv: &Uint<LIMBS>,
) -> Uint<LIMBS> {
    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    if LIMBS == bigint::WIDTH_WORDS {
        // In the RISC Zero zkVM 256-bit residues are represented in standard form instead of
        // Montgomery because, with the accelerator, multiplication is more efficient.
        return a.clone();
    }

    montgomery_reduction::<LIMBS>(&(*a, Uint::<LIMBS>::ZERO), modulus, mod_neg_inv)
}
