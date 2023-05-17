use crate::{modular::pow::pow_montgomery_form, PowBoundedExp, Uint};

use super::DynResidue;

impl<const LIMBS: usize> DynResidue<LIMBS> {
    /// Raises to the `exponent` power.
    pub fn pow(&self, exponent: &Uint<LIMBS>) -> DynResidue<LIMBS> {
        self.pow_bounded_exp(exponent, Uint::<LIMBS>::BITS)
    }

    /// Raises to the `exponent` power,
    /// with `exponent_bits` representing the number of (least significant) bits
    /// to take into account for the exponent.
    ///
    /// NOTE: `exponent_bits` may be leaked in the time pattern.
    pub fn pow_bounded_exp(&self, exponent: &Uint<LIMBS>, exponent_bits: usize) -> Self {
        Self {
            montgomery_form: pow_montgomery_form(
                &self.montgomery_form,
                exponent,
                exponent_bits,
                &self.residue_params.modulus,
                &self.residue_params.r,
                self.residue_params.mod_neg_inv,
            ),
            residue_params: self.residue_params,
        }
    }
}

impl<const LIMBS: usize> PowBoundedExp<Uint<LIMBS>> for DynResidue<LIMBS> {
    fn pow_bounded_exp(&self, exponent: &Uint<LIMBS>, exponent_bits: usize) -> Self {
        self.pow_bounded_exp(exponent, exponent_bits)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        modular::runtime_mod::{DynResidue, DynResidueParams},
        U256,
    };

    #[test]
    fn test_powmod_small_base() {
        let params = DynResidueParams::new(&U256::from_be_hex(
            "9CC24C5DF431A864188AB905AC751B727C9447A8E99E6366E1AD78A21E8D882B",
        ));

        let base = U256::from(105u64);
        let base_mod = DynResidue::new(&base, params);

        let exponent =
            U256::from_be_hex("77117F1273373C26C700D076B3F780074D03339F56DD0EFB60E7F58441FD3685");

        let res = base_mod.pow(&exponent);

        let expected =
            U256::from_be_hex("7B2CD7BDDD96C271E6F232F2F415BB03FE2A90BD6CCCEA5E94F1BFD064993766");
        assert_eq!(res.retrieve(), expected);
    }

    #[test]
    fn test_powmod_small_exponent() {
        let params = DynResidueParams::new(&U256::from_be_hex(
            "9CC24C5DF431A864188AB905AC751B727C9447A8E99E6366E1AD78A21E8D882B",
        ));

        let base =
            U256::from_be_hex("3435D18AA8313EBBE4D20002922225B53F75DC4453BB3EEC0378646F79B524A4");
        let base_mod = DynResidue::new(&base, params);

        let exponent = U256::from(105u64);

        let res = base_mod.pow(&exponent);

        let expected =
            U256::from_be_hex("89E2A4E99F649A5AE2C18068148C355CA927B34A3245C938178ED00D6EF218AA");
        assert_eq!(res.retrieve(), expected);
    }

    #[test]
    fn test_powmod_zero_exponent() {
        let params = DynResidueParams::new(&U256::from_be_hex(
            "9CC24C5DF431A864188AB905AC751B727C9447A8E99E6366E1AD78A21E8D882B",
        ));

        let base =
            U256::from_be_hex("3435D18AA8313EBBE4D20002922225B53F75DC4453BB3EEC0378646F79B524A4");
        let base_mod = DynResidue::new(&base, params);

        let res = base_mod.pow(&U256::ZERO);

        assert_eq!(res.retrieve(), U256::ONE);
    }

    #[test]
    fn test_powmod() {
        let params = DynResidueParams::new(&U256::from_be_hex(
            "9CC24C5DF431A864188AB905AC751B727C9447A8E99E6366E1AD78A21E8D882B",
        ));

        let base =
            U256::from_be_hex("3435D18AA8313EBBE4D20002922225B53F75DC4453BB3EEC0378646F79B524A4");
        let base_mod = DynResidue::new(&base, params);

        let exponent =
            U256::from_be_hex("77117F1273373C26C700D076B3F780074D03339F56DD0EFB60E7F58441FD3685");

        let res = base_mod.pow(&exponent);

        let expected =
            U256::from_be_hex("3681BC0FEA2E5D394EB178155A127B0FD2EF405486D354251C385BDD51B9D421");
        assert_eq!(res.retrieve(), expected);
    }
}
