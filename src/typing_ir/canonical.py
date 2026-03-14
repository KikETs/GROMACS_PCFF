from __future__ import annotations

import math


def _bond_label(bond: dict) -> tuple[str, str]:
    order = bond["order"]
    if order is None:
        normalized = "unknown"
    else:
        normalized = str(order)
    return normalized, bond["bond_code"]


def _distance_signature(atom: dict, atoms: list[dict]) -> tuple[float, ...]:
    distances = []
    x0, y0, z0 = atom["coordinates"]
    for other in atoms:
        x1, y1, z1 = other["coordinates"]
        distance = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)
        distances.append(round(distance, 8))
    return tuple(sorted(distances))


def _sortable_formal_charge(charge: int | None) -> tuple[int, int]:
    if charge is None:
        return (1, 0)
    return (0, charge)


def refine_atom_order(atoms: list[dict], bonds: list[dict]) -> list[dict]:
    adjacency: dict[int, list[tuple[int, tuple[str, str]]]] = {
        atom["source_index"]: [] for atom in atoms
    }
    for bond in bonds:
        left, right = bond["atom_source_indices"]
        label = _bond_label(bond)
        adjacency[left].append((right, label))
        adjacency[right].append((left, label))

    atom_by_index = {atom["source_index"]: atom for atom in atoms}
    labels: dict[int, tuple] = {}
    for atom in atoms:
        source_index = atom["source_index"]
        labels[source_index] = (
            atom["element"],
            _sortable_formal_charge(atom["formal_charge"]),
            len(adjacency[source_index]),
            tuple(sorted(label for _, label in adjacency[source_index])),
        )

    changed = True
    while changed:
        signatures = {}
        for atom in atoms:
            source_index = atom["source_index"]
            signatures[source_index] = (
                labels[source_index],
                tuple(
                    sorted(
                        (edge_label, labels[neighbor_index])
                        for neighbor_index, edge_label in adjacency[source_index]
                    )
                ),
            )

        ordered_signatures = sorted(
            {signature for signature in signatures.values()},
        )
        signature_to_rank = {
            signature: rank for rank, signature in enumerate(ordered_signatures, start=1)
        }
        new_labels = {
            source_index: signature_to_rank[signature]
            for source_index, signature in signatures.items()
        }
        changed = new_labels != labels
        labels = new_labels

    order = sorted(
        atoms,
        key=lambda atom: (
            labels[atom["source_index"]],
            _distance_signature(atom, atoms),
            tuple(round(value, 8) for value in atom["coordinates"]),
            atom["source_atom_id"],
            atom["source_index"],
        ),
    )
    return order
