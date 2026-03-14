from __future__ import annotations


ANGSTROM_TO_NM = 0.1
KCAL_TO_KJ = 4.184

ELEMENT_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Li": 6.941,
    "S": 32.065,
}

MOLECULE_NAME_BY_FAMILY = {
    "acyclic_alkane": "ALK",
    "acyclic_ether": "ETH",
    "lithium_cation": "LI",
    "tfsi_like_sulfonimide": "TFSI",
}


def format_float(value: float) -> str:
    return f"{value:.8f}"


def kcal_to_kj(value: float) -> float:
    return value * KCAL_TO_KJ


def angstrom_to_nm(value: float) -> float:
    return value * ANGSTROM_TO_NM


def bond_k2_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / (ANGSTROM_TO_NM**2)


def bond_k3_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / (ANGSTROM_TO_NM**3)


def bond_k4_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / (ANGSTROM_TO_NM**4)


def bond_bond_k_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / (ANGSTROM_TO_NM**2)


def bond_angle_k_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / ANGSTROM_TO_NM


def dihedral_bond_torsion_k_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / ANGSTROM_TO_NM


def molecule_name(component_family: str) -> str:
    if component_family in MOLECULE_NAME_BY_FAMILY:
        return MOLECULE_NAME_BY_FAMILY[component_family]
    sanitized = "".join(ch for ch in component_family.upper() if ch.isalnum())
    return (sanitized or "MOL")[:8]


def residue_name(component_family: str) -> str:
    return molecule_name(component_family)


def atom_name(element: str, ordinal: int) -> str:
    return f"{element}{ordinal}"[:5]
