from __future__ import annotations

import pytest
from tools.pcff_fixture_bridge.common import (
    kcal_to_kj,
    angstrom_to_nm,
    bond_k2_to_gromacs,
    bond_k3_to_gromacs,
    bond_k4_to_gromacs,
    bond_bond_k_to_gromacs,
    bond_angle_k_to_gromacs,
    dihedral_bond_torsion_k_to_gromacs,
)

def test_kcal_to_kj():
    # 1 kcal = 4.184 kJ
    assert kcal_to_kj(1.0) == pytest.approx(4.184)
    assert kcal_to_kj(100.0) == pytest.approx(418.4)

def test_angstrom_to_nm():
    # 1 A = 0.1 nm
    assert angstrom_to_nm(1.0) == pytest.approx(0.1)
    assert angstrom_to_nm(15.3) == pytest.approx(1.53)

def test_bond_k2_to_gromacs():
    # k2: kcal/mol/A^2 -> kJ/mol/nm^2
    # 250 kcal/mol/A^2 * 4.184 kJ/kcal / (0.1 nm/A)^2 = 104600
    assert bond_k2_to_gromacs(250.0) == pytest.approx(104600.0)

def test_bond_k3_to_gromacs():
    # k3: kcal/mol/A^3 -> kJ/mol/nm^3
    # -35 kcal/mol/A^3 * 4.184 / (0.1^3) = -146440
    assert bond_k3_to_gromacs(-35.0) == pytest.approx(-146440.0)

def test_bond_k4_to_gromacs():
    # k4: kcal/mol/A^4 -> kJ/mol/nm^4
    # 8.0 * 4.184 / (0.1^4) = 334720
    assert bond_k4_to_gromacs(8.0) == pytest.approx(334720.0)

def test_bond_bond_k_to_gromacs():
    # bb: kcal/mol/A^2 -> kJ/mol/nm^2
    assert bond_bond_k_to_gromacs(10.0) == pytest.approx(4184.0)

def test_bond_angle_k_to_gromacs():
    # ba: kcal/mol/A -> kJ/mol/nm
    # 5.0 * 4.184 / 0.1 = 209.2
    assert bond_angle_k_to_gromacs(5.0) == pytest.approx(209.2)

def test_dihedral_bond_torsion_k_to_gromacs():
    # mbt: kcal/mol/A -> kJ/mol/nm
    assert dihedral_bond_torsion_k_to_gromacs(2.0) == pytest.approx(83.68)
