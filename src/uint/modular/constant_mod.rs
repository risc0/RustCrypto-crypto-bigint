use core::{fmt::Debug, marker::PhantomData};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::{Limb, Uint, Zero};

use super::{div_by_2::div_by_2, reduction::montgomery_reduction, Retrieve};

#[cfg(feature = "rand_core")]
use crate::{rand_core::CryptoRngCore, NonZero, Random, RandomMod};

#[cfg(feature = "serde")]
use {
    crate::Encoding,
    serdect::serde::de::Error,
    serdect::serde::{Deserialize, Deserializer, Serialize, Serializer},
};

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use crate::risc0;

/// Additions between residues with a constant modulus
mod const_add;
/// Multiplicative inverses of residues with a constant modulus
mod const_inv;
/// Multiplications between residues with a constant modulus
mod const_mul;
/// Negations of residues with a constant modulus
mod const_neg;
/// Exponentiation of residues with a constant modulus
mod const_pow;
/// Subtractions between residues with a constant modulus
mod const_sub;

/// Macros to remove the boilerplate code when dealing with constant moduli.
#[macro_use]
mod macros;

pub use macros::*;

/// The parameters to efficiently go to and from the Montgomery form for a given odd modulus. An easy way to generate these parameters is using the `impl_modulus!` macro. These parameters are constant, so they cannot be set at runtime.
///
/// Unfortunately, `LIMBS` must be generic for now until const generics are stabilized.
pub trait ResidueParams<const LIMBS: usize>:
    Copy + Debug + Default + Eq + Send + Sync + 'static
{
    /// Number of limbs required to encode a residue
    const LIMBS: usize;

    /// The constant modulus
    const MODULUS: Uint<LIMBS>;
    /// Parameter used in Montgomery reduction
    const R: Uint<LIMBS>;
    /// R^2, used to move into Montgomery form
    const R2: Uint<LIMBS>;
    /// R^3, used to perform a multiplicative inverse
    const R3: Uint<LIMBS>;
    /// The lowest limbs of -(MODULUS^-1) mod R
    // We only need the LSB because during reduction this value is multiplied modulo 2**Limb::BITS.
    const MOD_NEG_INV: Limb;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A residue mod `MOD`, represented using `LIMBS` limbs. The modulus of this residue is constant, so it cannot be set at runtime.
pub struct Residue<MOD, const LIMBS: usize>
where
    MOD: ResidueParams<LIMBS>,
{
    montgomery_form: Uint<LIMBS>,
    phantom: PhantomData<MOD>,
}

#[cfg(feature = "zeroize")]
impl<MOD: ResidueParams<LIMBS>, const LIMBS: usize> zeroize::DefaultIsZeroes
    for Residue<MOD, LIMBS>
{
}

impl<MOD: ResidueParams<LIMBS>, const LIMBS: usize> Residue<MOD, LIMBS> {
    /// The representation of 0 mod `MOD`.
    pub const ZERO: Self = Self {
        montgomery_form: Uint::<LIMBS>::ZERO,
        phantom: PhantomData,
    };

    /// The representation of 1 mod `MOD`.
    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    pub const ONE: Self = {
        if LIMBS == risc0::BIGINT_WIDTH_WORDS {
            Self {
                montgomery_form: Uint::<LIMBS>::ONE,
                phantom: PhantomData,
            }
        } else {
            Self {
                montgomery_form: MOD::R,
                phantom: PhantomData,
            }
        }
    };

    /// The representation of 1 mod `MOD`.
    #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
    pub const ONE: Self = Self {
        montgomery_form: MOD::R,
        phantom: PhantomData,
    };

    /// Instantiates a new `Residue` that represents this `integer` mod `MOD`.
    pub fn new(integer: &Uint<LIMBS>) -> Self {
        #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
        if LIMBS == risc0::BIGINT_WIDTH_WORDS {
            // When working with U256 in the RISC Zero zkVM, leave the value in standard form.
            // Ensure that the input is reduced by passing it though a modmul by one.
            return Self {
                montgomery_form: risc0::modmul_uint_256(
                    &integer,
                    &Uint::<LIMBS>::ONE,
                    &MOD::MODULUS,
                ),
                phantom: PhantomData,
            };
        }

        let product = integer.mul_wide(&MOD::R2);
        let montgomery_form =
            montgomery_reduction::<LIMBS>(&product, &MOD::MODULUS, MOD::MOD_NEG_INV);

        Self {
            montgomery_form,
            phantom: PhantomData,
        }
    }

    /// Retrieves the integer currently encoded in this `Residue`, guaranteed to be reduced.
    pub fn retrieve(&self) -> Uint<LIMBS> {
        #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
        if LIMBS == risc0::BIGINT_WIDTH_WORDS {
            // In the RISC Zero zkVM 256-bit residues are represented in standard form.
            return self.montgomery_form;
        }

        montgomery_reduction::<LIMBS>(
            &(self.montgomery_form, Uint::ZERO),
            &MOD::MODULUS,
            MOD::MOD_NEG_INV,
        )
    }

    /// Performs the modular division by 2, that is for given `x` returns `y`
    /// such that `y * 2 = x mod p`. This means:
    /// - if `x` is even, returns `x / 2`,
    /// - if `x` is odd, returns `(x + p) / 2`
    ///   (since the modulus `p` in Montgomery form is always odd, this divides entirely).
    pub fn div_by_2(&self) -> Self {
        Self {
            montgomery_form: div_by_2(&self.montgomery_form, &MOD::MODULUS),
            phantom: PhantomData,
        }
    }
}

impl<MOD: ResidueParams<LIMBS> + Copy, const LIMBS: usize> ConditionallySelectable
    for Residue<MOD, LIMBS>
{
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Residue {
            montgomery_form: Uint::conditional_select(
                &a.montgomery_form,
                &b.montgomery_form,
                choice,
            ),
            phantom: PhantomData,
        }
    }
}

impl<MOD: ResidueParams<LIMBS>, const LIMBS: usize> ConstantTimeEq for Residue<MOD, LIMBS> {
    fn ct_eq(&self, other: &Self) -> Choice {
        ConstantTimeEq::ct_eq(&self.montgomery_form, &other.montgomery_form)
    }
}

impl<MOD: ResidueParams<LIMBS>, const LIMBS: usize> Default for Residue<MOD, LIMBS> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<MOD: ResidueParams<LIMBS>, const LIMBS: usize> Zero for Residue<MOD, LIMBS> {
    const ZERO: Self = Self::ZERO;
}

#[cfg(feature = "rand_core")]
impl<MOD, const LIMBS: usize> Random for Residue<MOD, LIMBS>
where
    MOD: ResidueParams<LIMBS>,
{
    #[inline]
    fn random(rng: &mut impl CryptoRngCore) -> Self {
        Self::new(&Uint::random_mod(rng, &NonZero::from_uint(MOD::MODULUS)))
    }
}

impl<MOD: ResidueParams<LIMBS>, const LIMBS: usize> Retrieve for Residue<MOD, LIMBS> {
    type Output = Uint<LIMBS>;
    fn retrieve(&self) -> Self::Output {
        self.retrieve()
    }
}

#[cfg(feature = "serde")]
impl<'de, MOD, const LIMBS: usize> Deserialize<'de> for Residue<MOD, LIMBS>
where
    MOD: ResidueParams<LIMBS>,
    Uint<LIMBS>: Encoding,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
        let r_inv: Uint<{ LIMBS }> = MOD::R.inv_odd_mod(&MOD::MODULUS).0;

        Uint::<LIMBS>::deserialize(deserializer).and_then(|montgomery_form| {
            if Uint::ct_lt(&montgomery_form, &MOD::MODULUS).into() {
                #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
                if LIMBS == risc0::BIGINT_WIDTH_WORDS {
                    // In the RISC Zero zkVM 256-bit residues are represented in standard form.
                    // To ensure this is interoperable with the host, convert to standard form.
                    let value = risc0::modmul_uint_256(&montgomery_form, &r_inv, &MOD::MODULUS);
                    return Ok(Self {
                        montgomery_form: value,
                        phantom: PhantomData,
                    });
                }

                Ok(Self {
                    montgomery_form,
                    phantom: PhantomData,
                })
            } else {
                Err(D::Error::custom("montgomery form must be reduced"))
            }
        })
    }
}

#[cfg(feature = "serde")]
impl<MOD, const LIMBS: usize> Serialize for Residue<MOD, LIMBS>
where
    MOD: ResidueParams<LIMBS>,
    Uint<LIMBS>: Encoding,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
        if LIMBS == risc0::BIGINT_WIDTH_WORDS {
            // In the RISC Zero zkVM 256-bit residues are represented in standard form.
            // To ensure this is interoperable with the host, convert to Montgomery form.
            let value = risc0::modmul_uint_256(&self.montgomery_form, &MOD::R, &MOD::MODULUS);
            return value.serialize(serializer);
        }

        self.montgomery_form.serialize(serializer)
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use crate::{const_residue, impl_modulus, modular::constant_mod::ResidueParams, U256};

    impl_modulus!(
        Modulus,
        U256,
        "9CC24C5DF431A864188AB905AC751B727C9447A8E99E6366E1AD78A21E8D882B"
    );

    #[test]
    fn serde_roundtrip() {
        let value_uint = U256::from(105u64);
        let value = const_residue!(value_uint, Modulus);

        let value_encoded = bincode::serialize(&value).unwrap();
        let value_decoded = bincode::deserialize(&value_encoded).unwrap();

        assert_eq!(value, value_decoded);
    }
}
