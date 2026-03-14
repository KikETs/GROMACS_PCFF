from __future__ import annotations

import math


def canonical_family_pair(family_a: str, family_b: str) -> tuple[list[str], str]:
    ordered = sorted((family_a, family_b))
    return ordered, format_family_pair(ordered[0], ordered[1])


def format_family_pair(family_a: str, family_b: str) -> str:
    return f"pair({family_a}|{family_b})"


def sixthpower_mix(
    sigma_a_angstrom: float,
    epsilon_a_kcal_mol: float,
    sigma_b_angstrom: float,
    epsilon_b_kcal_mol: float,
) -> dict:
    sigma_a6 = sigma_a_angstrom**6
    sigma_b6 = sigma_b_angstrom**6
    sigma_mixed = ((sigma_a6 + sigma_b6) / 2.0) ** (1.0 / 6.0)
    epsilon_mixed = (
        2.0
        * math.sqrt(epsilon_a_kcal_mol * epsilon_b_kcal_mol)
        * (sigma_a_angstrom**3)
        * (sigma_b_angstrom**3)
        / (sigma_a6 + sigma_b6)
    )
    return {
        "sigma_angstrom": round(sigma_mixed, 8),
        "epsilon_kcal_mol": round(epsilon_mixed, 8),
    }


def class2_normal_coefficients(epsilon_kcal_mol: float, sigma_angstrom: float) -> dict:
    return {
        "c6_kcal_mol_angstrom6": round(18.0 * epsilon_kcal_mol * (sigma_angstrom**6), 8),
        "c9_kcal_mol_angstrom9": round(18.0 * epsilon_kcal_mol * (sigma_angstrom**9), 8),
        "dispersion_power": 6,
        "repulsion_power": 9,
    }


def class2_pair14_coefficients(epsilon_kcal_mol: float, sigma_angstrom: float) -> dict:
    return {
        "c6_kcal_mol_angstrom6": round(3.0 * epsilon_kcal_mol * (sigma_angstrom**6), 8),
        "c9_kcal_mol_angstrom9": round(2.0 * epsilon_kcal_mol * (sigma_angstrom**9), 8),
        "dispersion_power": 6,
        "repulsion_power": 9,
    }
