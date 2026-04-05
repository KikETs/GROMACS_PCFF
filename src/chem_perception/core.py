from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import re

from typing_ir import dumps_ir, load_ir, parse_file, validate_ir

from .errors import SchemaError


SCHEMA_NAME = "chem_perception"
SCHEMA_VERSION = 1
RING_MODEL = "edge_shortest_cycle_v1"
AROMATICITY_MODEL = "explicit_and_kekule_huckel_v1"
VALENCE_MODEL = "local_explicit_bond_sum_v1"
POLYMER_TAG_MODEL = "annotation_or_dummy_placeholder_v1"

PLACEHOLDER_ELEMENT_RE = re.compile(r"^(Du|R\d*|\*)$")
LONE_PAIR_DONOR_ELEMENTS = {"N", "O", "S", "P"}

ALLOWED_VALENCES = {
    ("H", -1): (0,),
    ("H", 0): (1,),
    ("C", -1): (3,),
    ("C", 0): (4,),
    ("C", 1): (3,),
    ("N", -1): (2,),
    ("N", 0): (3, 5),
    ("N", 1): (4,),
    ("O", -1): (1,),
    ("O", 0): (2,),
    ("O", 1): (3,),
    ("F", -1): (0,),
    ("F", 0): (1,),
    ("Li", 1): (0,),
    ("P", 0): (3, 5),
    ("P", 1): (4, 6),
    ("S", -1): (1, 3, 5),
    ("S", 0): (2, 4, 6),
    ("S", 1): (3, 5),
}


def perceive_file(
    path: str | Path,
    *,
    input_format: str | None = None,
    source_id: str | None = None,
) -> dict:
    return perceive_ir(
        parse_file(path, input_format=input_format, source_id=source_id),
    )


def perceive_ir(ir: dict) -> dict:
    validate_ir(ir)
    component = ir["components"][0]
    graph = _build_graph(component)
    ring_data = _compute_ring_data(graph)
    atom_ring_map = _atom_ring_map(ring_data["rings"], graph["atom_indices"])
    bond_ring_map = _bond_ring_map(ring_data["rings"], graph["bond_indices"])
    aromaticity = _compute_aromaticity(graph, ring_data["rings"], atom_ring_map, bond_ring_map)
    polymer_tags = _compute_polymer_connection_tags(graph)
    atom_features = _build_atom_features(graph, atom_ring_map, aromaticity, polymer_tags)
    bond_features = _build_bond_features(graph, bond_ring_map, aromaticity)
    aromatic_systems = _compute_aromatic_systems(graph, aromaticity["aromatic_bond_indices"])

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "typed_ir_schema_name": ir["schema_name"],
            "typed_ir_schema_version": ir["schema_version"],
            "typed_ir_sha256": hashlib.sha256(dumps_ir(ir).encode("utf-8")).hexdigest(),
            "source_id": ir["source"]["source_id"],
            "input_format": ir["source"]["input_format"],
        },
        "perception": {
            "status": "computed",
            "ring_model": RING_MODEL,
            "aromaticity_model": AROMATICITY_MODEL,
            "valence_model": VALENCE_MODEL,
            "polymer_tag_model": POLYMER_TAG_MODEL,
        },
        "components": [
            {
                "component_id": component["component_id"],
                "name": component["name"],
                "atom_count": component["atom_count"],
                "bond_count": component["bond_count"],
                "atoms": atom_features,
                "bonds": bond_features,
                "rings": ring_data["rings"],
                "ring_systems": ring_data["ring_systems"],
                "aromatic_systems": aromatic_systems,
                "polymer_connection_points": polymer_tags["points"],
            }
        ],
    }
    validate_report(report)
    return report


def dumps_report(report: dict) -> str:
    validate_report(report)
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_report(report), encoding="utf-8")


def loads_report(text: str) -> dict:
    report = json.loads(text)
    validate_report(report)
    return report


def load_report(path: Path) -> dict:
    return loads_report(path.read_text(encoding="utf-8"))


def validate_report(report: dict) -> None:
    if report.get("schema_name") != SCHEMA_NAME:
        raise SchemaError("invalid_perception_schema", "schema_name must be 'chem_perception'")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError("invalid_perception_schema", "Unsupported schema_version")

    source = report.get("source")
    if not isinstance(source, dict):
        raise SchemaError("invalid_perception_schema", "source must be a mapping")
    for key in {
        "typed_ir_schema_name",
        "typed_ir_schema_version",
        "typed_ir_sha256",
        "source_id",
        "input_format",
    }:
        if key not in source:
            raise SchemaError("invalid_perception_schema", f"source.{key} is required")

    components = report.get("components")
    if not isinstance(components, list) or len(components) != 1:
        raise SchemaError("invalid_perception_schema", "PT2 perception report must contain exactly one component")

    component = components[0]
    atoms = component.get("atoms")
    bonds = component.get("bonds")
    rings = component.get("rings")
    if not isinstance(atoms, list) or not isinstance(bonds, list) or not isinstance(rings, list):
        raise SchemaError("invalid_perception_schema", "component atoms, bonds, and rings must be lists")
    if component.get("atom_count") != len(atoms):
        raise SchemaError("invalid_perception_schema", "component atom_count mismatch")
    if component.get("bond_count") != len(bonds):
        raise SchemaError("invalid_perception_schema", "component bond_count mismatch")

    for expected_index, atom in enumerate(atoms, start=1):
        if atom.get("canonical_index") != expected_index:
            raise SchemaError("invalid_perception_schema", "Atom canonical indices must be contiguous")
        for key in {
            "element",
            "neighbor_indices",
            "valence",
            "ring",
            "aromaticity",
            "coordination",
            "improper_center_candidate",
            "polymer_connection",
        }:
            if key not in atom:
                raise SchemaError("invalid_perception_schema", f"Atom field {key!r} is required")

    for expected_index, bond in enumerate(bonds, start=1):
        if bond.get("canonical_index") != expected_index:
            raise SchemaError("invalid_perception_schema", "Bond canonical indices must be contiguous")
        for key in {"atom_indices", "ring", "aromaticity"}:
            if key not in bond:
                raise SchemaError("invalid_perception_schema", f"Bond field {key!r} is required")


def _build_graph(component: dict) -> dict:
    atoms = {atom["canonical_index"]: atom for atom in component["atoms"]}
    bonds = {bond["canonical_index"]: bond for bond in component["bonds"]}
    adjacency: dict[int, set[int]] = {index: set() for index in atoms}
    pair_to_bond_index: dict[tuple[int, int], int] = {}
    atom_to_bond_indices: dict[int, list[int]] = {index: [] for index in atoms}

    for bond in component["bonds"]:
        left, right = bond["atom_indices"]
        pair = tuple(sorted((left, right)))
        adjacency[left].add(right)
        adjacency[right].add(left)
        pair_to_bond_index[pair] = bond["canonical_index"]
        atom_to_bond_indices[left].append(bond["canonical_index"])
        atom_to_bond_indices[right].append(bond["canonical_index"])

    for index in atom_to_bond_indices:
        atom_to_bond_indices[index].sort()

    return {
        "component": component,
        "atoms": atoms,
        "bonds": bonds,
        "adjacency": {index: sorted(neighbors) for index, neighbors in adjacency.items()},
        "pair_to_bond_index": pair_to_bond_index,
        "atom_to_bond_indices": atom_to_bond_indices,
        "atom_indices": sorted(atoms),
        "bond_indices": sorted(bonds),
    }


def _compute_ring_data(graph: dict) -> dict:
    cycle_map: dict[tuple[int, ...], dict] = {}
    ring_bond_indices = set()
    ring_bond_graph: dict[int, set[int]] = {index: set() for index in graph["atom_indices"]}

    for bond_index, bond in graph["bonds"].items():
        left, right = bond["atom_indices"]
        blocked_pair = tuple(sorted((left, right)))
        path = _shortest_path(graph["adjacency"], left, right, blocked_pair)
        if path is None:
            continue

        cycle_key = _canonical_cycle(path)
        cycle_bond_indices = []
        cycle_atoms = list(cycle_key)
        for offset, atom_index in enumerate(cycle_atoms):
            next_index = cycle_atoms[(offset + 1) % len(cycle_atoms)]
            pair = tuple(sorted((atom_index, next_index)))
            cycle_bond_index = graph["pair_to_bond_index"][pair]
            cycle_bond_indices.append(cycle_bond_index)
            ring_bond_indices.add(cycle_bond_index)
            ring_bond_graph[pair[0]].add(pair[1])
            ring_bond_graph[pair[1]].add(pair[0])

        cycle_map[cycle_key] = {
            "ring_id": "",
            "atom_indices": cycle_atoms,
            "bond_indices": sorted(cycle_bond_indices),
            "size": len(cycle_atoms),
        }

    ring_systems = []
    ring_system_membership: dict[int, str] = {}
    visited = set()
    ring_atoms = {atom_index for atom_index, neighbors in ring_bond_graph.items() if neighbors}
    system_counter = 0
    for start in sorted(ring_atoms):
        if start in visited:
            continue
        system_counter += 1
        queue = [start]
        system_atoms = set()
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            system_atoms.add(current)
            queue.extend(neighbor for neighbor in sorted(ring_bond_graph[current]) if neighbor not in visited)
        ring_system_id = f"ring_system_{system_counter}"
        bond_indices = sorted(
            graph["pair_to_bond_index"][tuple(sorted((left, right)))]
            for left in system_atoms
            for right in ring_bond_graph[left]
            if left < right
        )
        for atom_index in system_atoms:
            ring_system_membership[atom_index] = ring_system_id
        ring_systems.append(
            {
                "ring_system_id": ring_system_id,
                "atom_indices": sorted(system_atoms),
                "bond_indices": bond_indices,
            }
        )

    rings = []
    for ring_counter, cycle_key in enumerate(sorted(cycle_map), start=1):
        ring = dict(cycle_map[cycle_key])
        ring["ring_id"] = f"ring_{ring_counter}"
        ring["ring_system_id"] = ring_system_membership[ring["atom_indices"][0]]
        rings.append(ring)

    return {
        "rings": rings,
        "ring_systems": ring_systems,
    }


def _shortest_path(
    adjacency: dict[int, list[int]],
    start: int,
    goal: int,
    blocked_pair: tuple[int, int],
) -> list[int] | None:
    queue = deque([start])
    parents = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in adjacency[current]:
            if tuple(sorted((current, neighbor))) == blocked_pair:
                continue
            if neighbor in parents:
                continue
            parents[neighbor] = current
            queue.append(neighbor)
    if goal not in parents:
        return None

    path = [goal]
    current = goal
    while parents[current] is not None:
        current = parents[current]
        path.append(current)
    return list(reversed(path))


def _canonical_cycle(nodes: list[int]) -> tuple[int, ...]:
    base = tuple(nodes)
    rotations = []
    for candidate in (base, tuple(reversed(base))):
        for offset in range(len(candidate)):
            rotations.append(candidate[offset:] + candidate[:offset])
    return min(rotations)


def _atom_ring_map(rings: list[dict], atom_indices: list[int]) -> dict[int, list[dict]]:
    ring_map = {index: [] for index in atom_indices}
    for ring in rings:
        for atom_index in ring["atom_indices"]:
            ring_map[atom_index].append(ring)
    return ring_map


def _bond_ring_map(rings: list[dict], bond_indices: list[int]) -> dict[int, list[dict]]:
    ring_map = {index: [] for index in bond_indices}
    for ring in rings:
        for bond_index in ring["bond_indices"]:
            ring_map[bond_index].append(ring)
    return ring_map


def _compute_aromaticity(
    graph: dict,
    rings: list[dict],
    atom_ring_map: dict[int, list[dict]],
    bond_ring_map: dict[int, list[dict]],
) -> dict:
    ring_status_by_id: dict[str, dict] = {}
    aromatic_atom_indices = set()
    aromatic_bond_indices = set()
    indeterminate_atom_indices = set()
    indeterminate_bond_indices = set()

    for ring in rings:
        status = _evaluate_ring_aromaticity(graph, ring)
        ring_status_by_id[ring["ring_id"]] = status
        ring["aromaticity"] = {
            "status": status["status"],
            "electron_count": status["electron_count"],
            "reason": status["reason"],
        }
        if status["status"] == "aromatic":
            aromatic_atom_indices.update(ring["atom_indices"])
            aromatic_bond_indices.update(ring["bond_indices"])
        elif status["status"] == "indeterminate":
            indeterminate_atom_indices.update(ring["atom_indices"])
            indeterminate_bond_indices.update(ring["bond_indices"])

    atom_status = {}
    for atom_index, atom_rings in atom_ring_map.items():
        statuses = [ring_status_by_id[ring["ring_id"]]["status"] for ring in atom_rings]
        if "aromatic" in statuses:
            status = "aromatic"
            reason = "member_of_aromatic_ring"
        elif "indeterminate" in statuses:
            status = "indeterminate"
            reason = "ring_aromaticity_indeterminate"
        else:
            status = "non_aromatic"
            reason = "not_in_aromatic_ring"
        atom_status[atom_index] = {
            "status": status,
            "reason": reason,
            "aromatic_ring_ids": sorted(
                ring["ring_id"]
                for ring in atom_rings
                if ring_status_by_id[ring["ring_id"]]["status"] == "aromatic"
            ),
        }

    bond_status = {}
    for bond_index, bond_rings in bond_ring_map.items():
        statuses = [ring_status_by_id[ring["ring_id"]]["status"] for ring in bond_rings]
        bond = graph["bonds"][bond_index]
        if bond["bond_code"] == "ar":
            status = "aromatic"
            reason = "explicit_aromatic_bond_code"
        elif "aromatic" in statuses:
            status = "aromatic"
            reason = "member_of_aromatic_ring"
        elif "indeterminate" in statuses:
            status = "indeterminate"
            reason = "ring_aromaticity_indeterminate"
        else:
            status = "non_aromatic"
            reason = "not_in_aromatic_ring"
        bond_status[bond_index] = {
            "status": status,
            "reason": reason,
            "aromatic_ring_ids": sorted(
                ring["ring_id"]
                for ring in bond_rings
                if ring_status_by_id[ring["ring_id"]]["status"] == "aromatic"
            ),
        }

    return {
        "ring_status_by_id": ring_status_by_id,
        "atom_status": atom_status,
        "bond_status": bond_status,
        "aromatic_atom_indices": sorted(aromatic_atom_indices),
        "aromatic_bond_indices": sorted(aromatic_bond_indices),
        "indeterminate_atom_indices": sorted(indeterminate_atom_indices),
        "indeterminate_bond_indices": sorted(indeterminate_bond_indices),
    }


def _evaluate_ring_aromaticity(graph: dict, ring: dict) -> dict:
    ring_atoms = ring["atom_indices"]
    ring_atom_set = set(ring_atoms)
    ring_bonds = [graph["bonds"][bond_index] for bond_index in ring["bond_indices"]]

    if all(bond["bond_code"] == "ar" for bond in ring_bonds):
        return {
            "status": "aromatic",
            "electron_count": None,
            "reason": "explicit_aromatic_bond_code_cycle",
        }

    if any(bond["bond_code"] == "ar" for bond in ring_bonds):
        return {
            "status": "indeterminate",
            "electron_count": None,
            "reason": "mixed_aromatic_and_numeric_bond_encoding",
        }

    if any(bond["order"] is None for bond in ring_bonds):
        return {
            "status": "indeterminate",
            "electron_count": None,
            "reason": "missing_bond_orders",
        }

    electron_count = 0
    for atom_index in ring_atoms:
        atom = graph["atoms"][atom_index]
        cycle_neighbors = _cycle_neighbors(ring_atoms, atom_index)
        in_ring_orders = []
        for neighbor_index in cycle_neighbors:
            bond_index = graph["pair_to_bond_index"][tuple(sorted((atom_index, neighbor_index)))]
            order = graph["bonds"][bond_index]["order"]
            if order is None:
                return {
                    "status": "indeterminate",
                    "electron_count": None,
                    "reason": "missing_bond_orders",
                }
            in_ring_orders.append(order)

        exocyclic_multiple = False
        for neighbor_index in graph["adjacency"][atom_index]:
            if neighbor_index in cycle_neighbors:
                continue
            bond_index = graph["pair_to_bond_index"][tuple(sorted((atom_index, neighbor_index)))]
            bond = graph["bonds"][bond_index]
            if bond["order"] is None:
                return {
                    "status": "indeterminate",
                    "electron_count": None,
                    "reason": "missing_bond_orders",
                }
            if bond["order"] > 1 or bond["bond_code"] == "ar":
                exocyclic_multiple = True
                break

        double_count = sum(1 for order in in_ring_orders if order > 1)
        if double_count == 1:
            electron_count += 1
            continue
        if double_count > 1:
            return {
                "status": "non_aromatic",
                "electron_count": electron_count,
                "reason": "atom_has_multiple_in_ring_pi_bonds",
            }

        if all(order == 1 for order in in_ring_orders):
            if atom["element"] in LONE_PAIR_DONOR_ELEMENTS and not exocyclic_multiple and atom["formal_charge"] in {0, -1}:
                electron_count += 2
                continue
            if atom["element"] == "C" and atom["formal_charge"] == -1 and not exocyclic_multiple:
                electron_count += 2
                continue
            return {
                "status": "non_aromatic",
                "electron_count": electron_count,
                "reason": "ring_not_fully_conjugated",
            }

        return {
            "status": "non_aromatic",
            "electron_count": electron_count,
            "reason": "unsupported_ring_bond_pattern",
        }

    if electron_count <= 0 or electron_count % 4 != 2:
        return {
            "status": "non_aromatic",
            "electron_count": electron_count,
            "reason": "huckel_electron_count_mismatch",
        }

    return {
        "status": "aromatic",
        "electron_count": electron_count,
        "reason": "kekule_huckel_match",
    }


def _cycle_neighbors(ring_atoms: list[int], atom_index: int) -> list[int]:
    index = ring_atoms.index(atom_index)
    return [
        ring_atoms[index - 1],
        ring_atoms[(index + 1) % len(ring_atoms)],
    ]


def _compute_polymer_connection_tags(graph: dict) -> dict:
    placeholder_tags = {}
    points = []
    target_tags: dict[int, list[str]] = {index: [] for index in graph["atom_indices"]}
    target_sources: dict[int, list[str]] = {index: [] for index in graph["atom_indices"]}

    for atom_index in graph["atom_indices"]:
        atom = graph["atoms"][atom_index]
        explicit_label = atom["annotations"].get("polymer_connection_label")
        is_placeholder = explicit_label is not None or _is_placeholder_element(atom["element"])
        if not is_placeholder:
            continue

        tag = explicit_label or atom["atom_name"] or atom["element"] or f"placeholder_{atom_index}"
        placeholder_tags[atom_index] = {
            "tag": tag,
            "is_placeholder": True,
            "tag_source": "annotation" if explicit_label is not None else "placeholder_atom",
        }
        for neighbor_index in graph["adjacency"][atom_index]:
            target_tags[neighbor_index].append(tag)
            target_sources[neighbor_index].append(placeholder_tags[atom_index]["tag_source"])
            points.append(
                {
                    "tag": tag,
                    "placeholder_atom_index": atom_index,
                    "target_atom_index": neighbor_index,
                    "tag_source": placeholder_tags[atom_index]["tag_source"],
                }
            )

    return {
        "placeholder_tags": placeholder_tags,
        "target_tags": {
            atom_index: {
                "tags": sorted(target_tags[atom_index]),
                "tag_sources": sorted(set(target_sources[atom_index])),
            }
            for atom_index in graph["atom_indices"]
        },
        "points": sorted(
            points,
            key=lambda point: (point["target_atom_index"], point["tag"], point["placeholder_atom_index"]),
        ),
    }


def _is_placeholder_element(element: str) -> bool:
    return bool(PLACEHOLDER_ELEMENT_RE.match(element))


def _build_atom_features(
    graph: dict,
    atom_ring_map: dict[int, list[dict]],
    aromaticity: dict,
    polymer_tags: dict,
) -> list[dict]:
    features = []
    for atom_index in graph["atom_indices"]:
        atom = graph["atoms"][atom_index]
        neighbor_indices = list(graph["adjacency"][atom_index])
        heavy_neighbors = [index for index in neighbor_indices if graph["atoms"][index]["element"] != "H"]
        hydrogen_neighbors = [index for index in neighbor_indices if graph["atoms"][index]["element"] == "H"]

        neighbor_links = []
        for neighbor_index in neighbor_indices:
            bond_index = graph["pair_to_bond_index"][tuple(sorted((atom_index, neighbor_index)))]
            bond = graph["bonds"][bond_index]
            neighbor_links.append(
                {
                    "atom_index": neighbor_index,
                    "bond_index": bond_index,
                    "bond_code": bond["bond_code"],
                    "order": bond["order"],
                }
            )

        ring_sizes = sorted({ring["size"] for ring in atom_ring_map[atom_index]})
        ring_ids = sorted(ring["ring_id"] for ring in atom_ring_map[atom_index])
        ring_system_ids = sorted({ring["ring_system_id"] for ring in atom_ring_map[atom_index]})
        valence = _infer_valence(graph, atom_index)
        improper = _improper_center_candidacy(graph, atom_index, aromaticity["atom_status"][atom_index], valence)

        placeholder_info = polymer_tags["placeholder_tags"].get(atom_index)
        target_info = polymer_tags["target_tags"][atom_index]
        coordination = _coordination_environment(
            graph,
            atom_index,
            aromaticity["atom_status"][atom_index],
            valence,
            improper,
        )

        features.append(
            {
                "canonical_index": atom_index,
                "element": atom["element"],
                "formal_charge": atom["formal_charge"],
                "neighbor_indices": neighbor_indices,
                "neighbor_links": neighbor_links,
                "heavy_neighbor_indices": heavy_neighbors,
                "hydrogen_neighbor_indices": hydrogen_neighbors,
                "neighbor_element_counts": _element_histogram(graph, neighbor_indices),
                "valence": valence,
                "ring": {
                    "is_ring_atom": bool(atom_ring_map[atom_index]),
                    "ring_ids": ring_ids,
                    "ring_system_ids": ring_system_ids,
                    "ring_sizes": ring_sizes,
                    "smallest_ring_size": ring_sizes[0] if ring_sizes else None,
                },
                "aromaticity": aromaticity["atom_status"][atom_index],
                "coordination": coordination,
                "improper_center_candidate": improper,
                "polymer_connection": {
                    "is_placeholder": placeholder_info is not None,
                    "tags": (
                        [placeholder_info["tag"]]
                        if placeholder_info is not None
                        else target_info["tags"]
                    ),
                    "tag_sources": (
                        [placeholder_info["tag_source"]]
                        if placeholder_info is not None
                        else target_info["tag_sources"]
                    ),
                },
            }
        )
    return features


def _build_bond_features(
    graph: dict,
    bond_ring_map: dict[int, list[dict]],
    aromaticity: dict,
) -> list[dict]:
    features = []
    for bond_index in graph["bond_indices"]:
        bond = graph["bonds"][bond_index]
        ring_sizes = sorted({ring["size"] for ring in bond_ring_map[bond_index]})
        features.append(
            {
                "canonical_index": bond_index,
                "atom_indices": bond["atom_indices"],
                "order": bond["order"],
                "bond_code": bond["bond_code"],
                "ring": {
                    "is_ring_bond": bool(bond_ring_map[bond_index]),
                    "ring_ids": sorted(ring["ring_id"] for ring in bond_ring_map[bond_index]),
                    "ring_sizes": ring_sizes,
                    "smallest_ring_size": ring_sizes[0] if ring_sizes else None,
                },
                "aromaticity": aromaticity["bond_status"][bond_index],
            }
        )
    return features


def _element_histogram(graph: dict, atom_indices: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom_index in atom_indices:
        element = graph["atoms"][atom_index]["element"]
        counts.setdefault(element, 0)
        counts[element] += 1
    return dict(sorted(counts.items()))


def _infer_valence(graph: dict, atom_index: int) -> dict:
    atom = graph["atoms"][atom_index]
    if _is_placeholder_element(atom["element"]):
        return {
            "known": False,
            "allowed_valences": [],
            "explicit_bond_order_sum": None,
            "inferred_valence": None,
            "status": "not_modeled",
        }

    bond_orders = []
    has_unknown_order = False
    pi_bond_count = 0
    for bond_index in graph["atom_to_bond_indices"][atom_index]:
        bond = graph["bonds"][bond_index]
        if bond["order"] is None:
            has_unknown_order = True
        else:
            bond_orders.append(bond["order"])
            if bond["order"] > 1:
                pi_bond_count += 1
        if bond["bond_code"] == "ar":
            pi_bond_count += 1

    explicit_bond_order_sum = None if has_unknown_order else sum(bond_orders)
    if atom["formal_charge"] is None:
        status = "indeterminate_missing_formal_charge"
        allowed_valences = []
        inferred_valence = None
    else:
        allowed_valences = list(ALLOWED_VALENCES.get((atom["element"], atom["formal_charge"]), ()))
        if not allowed_valences:
            status = "not_modeled"
            inferred_valence = None
        elif explicit_bond_order_sum is None:
            status = "indeterminate_missing_bond_orders"
            inferred_valence = None
        elif explicit_bond_order_sum in allowed_valences:
            status = "exact"
            inferred_valence = explicit_bond_order_sum
        elif explicit_bond_order_sum < min(allowed_valences):
            status = "underfilled"
            inferred_valence = None
        elif explicit_bond_order_sum > max(allowed_valences):
            status = "overfilled"
            inferred_valence = None
        else:
            status = "ambiguous"
            inferred_valence = None

    return {
        "known": status == "exact",
        "allowed_valences": allowed_valences,
        "explicit_bond_order_sum": explicit_bond_order_sum,
        "inferred_valence": inferred_valence,
        "pi_bond_count": pi_bond_count,
        "status": status,
    }


def _coordination_environment(
    graph: dict,
    atom_index: int,
    aromaticity: dict,
    valence: dict,
    improper: dict,
) -> dict:
    neighbor_indices = graph["adjacency"][atom_index]
    heavy_neighbors = [index for index in neighbor_indices if graph["atoms"][index]["element"] != "H"]
    hydrogen_neighbors = [index for index in neighbor_indices if graph["atoms"][index]["element"] == "H"]
    degree = len(neighbor_indices)

    if degree == 0:
        geometry_hint = "monatomic"
    elif degree == 1:
        geometry_hint = "terminal"
    elif degree == 2:
        if aromaticity["status"] == "aromatic" or valence["pi_bond_count"] >= 2:
            geometry_hint = "linear_or_sp_candidate"
        else:
            geometry_hint = "bent_or_chain_candidate"
    elif degree == 3:
        if improper["is_candidate"] and "planar_trigonal" in improper["kinds"]:
            geometry_hint = "trigonal_planar_candidate"
        else:
            geometry_hint = "trigonal_pyramidal_candidate"
    elif degree == 4:
        if valence["known"] and valence["inferred_valence"] is not None and valence["inferred_valence"] >= 6:
            geometry_hint = "tetrahedral_hypervalent_candidate"
        else:
            geometry_hint = "tetrahedral_candidate"
    elif degree == 5:
        geometry_hint = "trigonal_bipyramidal_candidate"
    elif degree == 6:
        geometry_hint = "octahedral_candidate"
    else:
        geometry_hint = "hypercoordinate_candidate"

    return {
        "coordination_number": degree,
        "heavy_coordination_number": len(heavy_neighbors),
        "hydrogen_coordination_number": len(hydrogen_neighbors),
        "geometry_hint": geometry_hint,
    }


def _improper_center_candidacy(
    graph: dict,
    atom_index: int,
    aromaticity: dict,
    valence: dict,
) -> dict:
    neighbors = graph["adjacency"][atom_index]
    degree = len(neighbors)
    kinds = []
    ordered_neighbors = []

    if degree == 3:
        if aromaticity["status"] == "aromatic" or valence["pi_bond_count"] >= 1:
            kinds.append("planar_trigonal")
            ordered_neighbors = _ordered_neighbors_for_center(graph, atom_index)
    elif degree == 4:
        signatures = [
            _substituent_signature(graph, atom_index, neighbor_index, depth=3)
            for neighbor_index in neighbors
        ]
        if len(set(signatures)) == 4:
            kinds.append("tetrahedral_distinct_substituents")
            ordered_neighbors = _ordered_neighbors_for_center(graph, atom_index)

    return {
        "is_candidate": bool(kinds),
        "kinds": kinds,
        "ordered_neighbor_indices": ordered_neighbors,
    }


def _ordered_neighbors_for_center(graph: dict, atom_index: int) -> list[int]:
    return [
        neighbor_index
        for _, neighbor_index in sorted(
            [
                (
                    _substituent_signature(graph, atom_index, neighbor_index, depth=3),
                    neighbor_index,
                )
                for neighbor_index in graph["adjacency"][atom_index]
            ],
            key=lambda item: (_signature_sort_key(item[0]), item[1]),
        )
    ]


def _signature_sort_key(value: object) -> tuple:
    if isinstance(value, tuple):
        return (0, tuple(_signature_sort_key(item) for item in value))
    if value is None:
        return (1, "")
    if isinstance(value, bool):
        return (2, int(value))
    if isinstance(value, int):
        return (3, value)
    if isinstance(value, float):
        return (4, value)
    if isinstance(value, str):
        return (5, value)
    return (6, repr(value))


def _substituent_signature(
    graph: dict,
    center_index: int,
    atom_index: int,
    *,
    depth: int,
) -> tuple:
    atom = graph["atoms"][atom_index]
    if depth == 0:
        return (
            atom["element"],
            atom["formal_charge"],
            len(graph["adjacency"][atom_index]),
        )

    children = []
    for neighbor_index in graph["adjacency"][atom_index]:
        if neighbor_index == center_index:
            continue
        bond_index = graph["pair_to_bond_index"][tuple(sorted((atom_index, neighbor_index)))]
        bond = graph["bonds"][bond_index]
        children.append(
            (
                bond["bond_code"],
                bond["order"],
                _substituent_signature(graph, atom_index, neighbor_index, depth=depth - 1),
            )
        )
    return (
        atom["element"],
        atom["formal_charge"],
        tuple(sorted(children, key=_signature_sort_key)),
    )


def _compute_aromatic_systems(graph: dict, aromatic_bond_indices: list[int]) -> list[dict]:
    aromatic_pairs = {
        tuple(sorted(graph["bonds"][bond_index]["atom_indices"]))
        for bond_index in aromatic_bond_indices
    }
    adjacency: dict[int, set[int]] = {index: set() for index in graph["atom_indices"]}
    for left, right in aromatic_pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)

    systems = []
    visited = set()
    system_counter = 0
    for start in sorted(index for index, neighbors in adjacency.items() if neighbors):
        if start in visited:
            continue
        system_counter += 1
        queue = [start]
        atoms = set()
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            atoms.add(current)
            queue.extend(neighbor for neighbor in sorted(adjacency[current]) if neighbor not in visited)
        bond_indices = sorted(
            graph["pair_to_bond_index"][tuple(sorted((left, right)))]
            for left in atoms
            for right in adjacency[left]
            if left < right
        )
        systems.append(
            {
                "aromatic_system_id": f"aromatic_system_{system_counter}",
                "atom_indices": sorted(atoms),
                "bond_indices": bond_indices,
            }
        )
    return systems
