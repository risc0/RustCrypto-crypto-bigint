use crate::{modular::mul::mul_montgomery_form, CtChoice, Limb, Uint};

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use risc0_zkvm_platform::syscall::bigint;

pub fn inv_montgomery_form<const LIMBS: usize>(
    x: &Uint<LIMBS>,
    modulus: &Uint<LIMBS>,
    r3: &Uint<LIMBS>,
    mod_neg_inv: Limb,
    r_inv: &Uint<LIMBS>,
) -> (Uint<LIMBS>, CtChoice) {
    let (inverse, is_some) = x.inv_odd_mod(modulus);

    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    if LIMBS == bigint::WIDTH_WORDS {
        return (inverse, is_some);
    }

    (
        mul_montgomery_form(&inverse, &r3, modulus, mod_neg_inv, &r_inv),
        is_some,
    )
}
