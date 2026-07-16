"""Industrially-aware synthetic PLM metadata for the TMCAD demo dataset.

This module shows how to connect the HOOPS AI Context Layer to an external
metadata system (PLM, ERP, or any database). It is split into two parts so the
integration boundary is obvious:

* ``OnDemandContextProvider`` — the thin adapter you implement. It subclasses
  ``ContextProvider`` and implements only the three methods the Context Layer
  calls. In your own project this is where you translate those calls into your
  PLM/ERP client's read/write/schema endpoints.
* ``SyntheticPLM`` — a stand-in for that external system. It generates stable,
  hash-derived records so the tutorial is reproducible without a real backend.
  You would delete this class and point the adapter at your real client.

What makes this backend industrially aware: the TMCAD models are stored one
folder per part family (bearing, bolt, bracket, coupling, flange, gear, nut,
pulley, screw, shaft), and the id of every indexed part keeps that folder in its
path. Instead of inventing random materials and processes, the backend reads the
part family from the id and returns realistic manufacturing metadata for that
family — alloys a process engineer would actually choose, the primary machining
operation, the heat treatment (for families that undergo one), the full process
route, and a cost derived from those choices.

The gear family follows the process flow of a typical high-strength automotive
gear, where every stage drives accuracy, fatigue life, and cost:

    1. Prepare raw material (forged blank or bar stock)
    2. Rough machining (turning to set OD/ID/face)
    3. Gear cutting (hobbing, shaping)
    4. Heat treatment (carburizing, induction hardening, nitriding)
    5. Finish grinding (removes heat-treatment distortion)
    6. Inspection (tooth profile, lead, pitch error)

Use it like this::

    from database import OnDemandContextProvider

    provider = OnDemandContextProvider()        # wraps a SyntheticPLM backend
    metadata = provider.get_contexts([h.id for h in hits])
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from hoops_ai.ml.context_layer import ContextProvider

# Relative material cost index (roughly proportional to grade + alloy content).
# Used only to derive a plausible part cost; not a real price list.
MATERIAL_COST_INDEX: dict[str, float] = {
    "Mild Steel (S235JR)": 40,
    "Carbon Steel (C35)": 48,
    "Carbon Steel (C45)": 52,
    "Alloy Steel (42CrMo4)": 72,
    "Case-Hardening Steel (16MnCr5)": 82,
    "Case-Hardening Steel (20MnCr5)": 88,
    "Case-Hardening Steel (18CrNiMo7-6)": 112,
    "Bearing Steel (100Cr6)": 98,
    "Chrome Steel (AISI 52100)": 96,
    "Stainless Steel (AISI 304)": 120,
    "Stainless Steel (AISI 316)": 150,
    "Stainless Steel (AISI 440C)": 140,
    "Stainless Steel (A2-70)": 125,
    "Aluminum (6061-T6)": 95,
    "Aluminum (7075-T6)": 130,
    "Gray Cast Iron (EN-GJL-250)": 34,
    "Ductile Iron (EN-GJS-500)": 44,
    "Property Class 8.8 Steel": 60,
    "Property Class 10.9 Steel": 74,
    "Property Class 12.9 Steel": 86,
}

# Relative process multiplier (labor + machine complexity per unit).
PROCESS_MULTIPLIER: dict[str, float] = {
    "Bar Sawing": 0.5,
    "Cold Heading": 0.6,
    "Cold Forming": 0.6,
    "Casting": 0.6,
    "Thread Rolling": 0.7,
    "Drilling & Tapping": 0.7,
    "Sheet-Metal Forming": 0.8,
    "Forging": 0.9,
    "Turning": 0.9,
    "CNC Milling": 1.0,
    "Cylindrical Grinding": 1.1,
    "Hard Turning": 1.2,
    "Broaching": 1.2,
    "Hobbing": 1.3,
    "Gear Shaping": 1.4,
    "Superfinishing": 1.6,
    "5-Axis Machining": 1.8,
    "Gear Grinding": 1.9,
}

# Per-family manufacturing profile. ``materials`` and ``processes`` are weighted
# so a dominant grade / operation emerges (consensus) while realistic
# alternatives still appear. ``route`` is the process flow an expert would name
# for that family. ``feature_base`` / ``feature_span`` bound the family's
# machined-feature count (teeth for gears, holes for brackets, and so on).
CATEGORY_PROFILES: dict[str, dict] = {
    "bearing": {
        "materials": [
            ("Bearing Steel (100Cr6)", 60),
            ("Chrome Steel (AISI 52100)", 25),
            ("Stainless Steel (AISI 440C)", 15),
        ],
        "processes": [
            ("Hard Turning", 45),
            ("Cylindrical Grinding", 35),
            ("Superfinishing", 20),
        ],
        # Standard bearing steel is through-hardened; case-carburizing is reserved
        # for the higher-load / shock-duty variant.
        "heat_treatments": [
            ("Through-Hardening", 65),
            ("Case Carburizing", 35),
        ],
        "route": (
            "Forged/bar blank -> Turning of races -> Through-hardening -> "
            "Raceway grinding -> Superfinishing -> Assembly of rolling elements -> Inspection"
        ),
        "feature_base": 6,
        "feature_span": 10,
    },
    "bolt": {
        "materials": [
            ("Property Class 8.8 Steel", 50),
            ("Property Class 10.9 Steel", 35),
            ("Stainless Steel (A2-70)", 15),
        ],
        "processes": [
            ("Cold Heading", 55),
            ("Thread Rolling", 45),
        ],
        "heat_treatments": [("Quench & Temper", 100)],
        "route": (
            "Wire drawing -> Cold heading (head form) -> Thread rolling -> "
            "Quench & temper -> Zinc/phosphate coating -> Inspection"
        ),
        "feature_base": 5,
        "feature_span": 8,
    },
    "bracket": {
        "materials": [
            ("Mild Steel (S235JR)", 45),
            ("Aluminum (6061-T6)", 35),
            ("Stainless Steel (AISI 304)", 20),
        ],
        "processes": [
            ("Sheet-Metal Forming", 50),
            ("CNC Milling", 35),
            ("Drilling & Tapping", 15),
        ],
        "route": (
            "Sheet blanking -> Bending/forming -> Drilling & tapping -> "
            "Welding (if multi-part) -> Deburring -> Powder coat/anodize -> Inspection"
        ),
        "feature_base": 4,
        "feature_span": 10,
    },
    "coupling": {
        "materials": [
            ("Carbon Steel (C45)", 45),
            ("Alloy Steel (42CrMo4)", 35),
            ("Ductile Iron (EN-GJS-500)", 20),
        ],
        "processes": [
            ("Turning", 45),
            ("Broaching", 30),
            ("CNC Milling", 25),
        ],
        "heat_treatments": [
            ("Induction Hardening", 55),
            ("Quench & Temper", 45),
        ],
        "route": (
            "Forged/bar blank -> Turning bore & OD -> Keyway broaching -> "
            "Drilling set-screw holes -> Heat treatment -> Balancing -> Inspection"
        ),
        "feature_base": 4,
        "feature_span": 8,
    },
    "flange": {
        "materials": [
            ("Carbon Steel (C45)", 40),
            ("Stainless Steel (AISI 316)", 35),
            ("Alloy Steel (42CrMo4)", 25),
        ],
        "processes": [
            ("Forging", 40),
            ("Turning", 35),
            ("Drilling & Tapping", 25),
        ],
        "route": (
            "Forging blank -> Rough facing & boring -> Bolt-hole drilling -> "
            "Finish facing (raised face) -> Surface finish -> NDT/Inspection"
        ),
        "feature_base": 6,
        "feature_span": 8,
    },
    "gear": {
        "materials": [
            ("Case-Hardening Steel (20MnCr5)", 50),
            ("Case-Hardening Steel (18CrNiMo7-6)", 28),
            ("Case-Hardening Steel (16MnCr5)", 15),
            ("Alloy Steel (42CrMo4)", 7),
        ],
        # Scoped to how a high-strength, case-hardened automotive gear is actually
        # cut and finished. A gear can theoretically be made 10+ ways (forging,
        # casting, powder metallurgy, injection molding, EDM, additive, rolling, ...),
        # but those are for low-load or non-precision gears and are intentionally
        # left out here, so the aggregated result reads like a real spec sheet.
        "processes": [
            ("Hobbing", 50),
            ("Gear Grinding", 25),
            ("Gear Shaping", 18),
            ("Broaching", 7),
        ],
        # Koji's three options for a high-strength automotive gear: carburizing is
        # the default for the heaviest-load teeth, induction hardening trades some
        # core toughness for a faster/cheaper cycle, and nitriding suits gears that
        # cannot tolerate the carburizing furnace's distortion. Weighted heavily
        # towards carburizing (consistent with EXPERT_NOTES) so it wins a clear
        # majority across a typical neighborhood instead of splitting evenly.
        "heat_treatments": [
            ("Carburizing", 65),
            ("Induction Hardening", 20),
            ("Nitriding", 15),
        ],
        "route": (
            "Forged blank / bar stock -> Rough turning (OD/ID/face) -> "
            "Gear cutting (hobbing/shaping) -> Heat treatment (carburizing / "
            "induction hardening / nitriding) -> Finish grinding -> Inspection (tooth profile, lead, pitch)"
        ),
        "feature_base": 12,
        "feature_span": 30,
    },
    "nut": {
        "materials": [
            ("Property Class 8.8 Steel", 50),
            ("Stainless Steel (A2-70)", 30),
            ("Carbon Steel (C35)", 20),
        ],
        "processes": [
            ("Cold Forming", 55),
            ("Drilling & Tapping", 45),
        ],
        "heat_treatments": [("Quench & Temper", 100)],
        "route": (
            "Bar/wire -> Cold forming (blank) -> Piercing -> Tapping -> "
            "Heat treatment -> Surface coating -> Inspection"
        ),
        "feature_base": 4,
        "feature_span": 6,
    },
    "pulley": {
        "materials": [
            ("Gray Cast Iron (EN-GJL-250)", 45),
            ("Aluminum (6061-T6)", 35),
            ("Carbon Steel (C45)", 20),
        ],
        "processes": [
            ("Casting", 40),
            ("Turning", 40),
            ("CNC Milling", 20),
        ],
        "route": (
            "Casting / bar blank -> Turning bore & groove profile -> "
            "Keyway broaching -> Balancing -> Surface finish -> Inspection"
        ),
        "feature_base": 4,
        "feature_span": 8,
    },
    "screw": {
        "materials": [
            ("Property Class 12.9 Steel", 55),
            ("Stainless Steel (A2-70)", 30),
            ("Carbon Steel (C45)", 15),
        ],
        "processes": [
            ("Cold Heading", 55),
            ("Thread Rolling", 45),
        ],
        "heat_treatments": [("Quench & Temper", 100)],
        "route": (
            "Wire drawing -> Cold heading (head + recess) -> Thread rolling -> "
            "Quench & temper -> Surface coating -> Inspection"
        ),
        "feature_base": 5,
        "feature_span": 8,
    },
    "shaft": {
        # Flatter distributions: shafts are made from several interchangeable
        # grades and routings, so material/process rarely reach hard consensus.
        "materials": [
            ("Alloy Steel (42CrMo4)", 38),
            ("Carbon Steel (C45)", 34),
            ("Stainless Steel (AISI 304)", 28),
        ],
        "processes": [
            ("Turning", 40),
            ("Cylindrical Grinding", 32),
            ("CNC Milling", 28),
        ],
        "heat_treatments": [
            ("Induction Hardening", 60),
            ("Quench & Temper", 40),
        ],
        "route": (
            "Bar stock -> Rough turning -> Milling keyways/splines -> "
            "Induction hardening -> Cylindrical grinding -> Inspection"
        ),
        "feature_base": 5,
        "feature_span": 10,
    },
}

DEFAULT_PROFILE: dict = {
    "materials": [
        ("Carbon Steel (C45)", 50),
        ("Aluminum (6061-T6)", 30),
        ("Stainless Steel (AISI 304)", 20),
    ],
    "processes": [
        ("CNC Milling", 50),
        ("Turning", 30),
        ("Drilling & Tapping", 20),
    ],
    "route": "Stock preparation -> Machining -> Heat treatment -> Finishing -> Inspection",
    "feature_base": 5,
    "feature_span": 10,
}

# One-line "what an expert would say" note per family.
EXPERT_NOTES: dict[str, str] = {
    "gear": (
        "High-strength automotive gears are carburized alloy steels; every stage from "
        "blank through finish grinding drives tooth accuracy, fatigue life, and cost."
    ),
    "bearing": (
        "Bearing rings are through-hardened and ground to micron tolerances; raceway "
        "finish governs noise, friction, and service life."
    ),
    "shaft": (
        "Power-transmission shafts balance torsional strength with machinability; "
        "keyways and induction-hardened journals are the main cost drivers."
    ),
    "screw": (
        "Fasteners are cold-formed and thread-rolled at high volume; grade and coating "
        "dominate cost far more than geometry."
    ),
    "_default": (
        "Material grade, primary process, and feature count together set the "
        "manufacturing cost of the part."
    ),
}

# Short qualitative description of the geometry an engineer would read off the
# part image. Kept at the family level so it stays truthful without claiming a
# per-part measurement the mock backend does not actually make.
CATEGORY_CHARACTERISTICS: dict[str, str] = {
    "bearing": "Concentric ground rings with rolling elements; precise bore and raceways.",
    "bolt": "Externally threaded shank with a formed hex or socket head.",
    "bracket": "L-shaped mounting plate with bores, oblong slots, and gusset ribs.",
    "coupling": "Cylindrical hub with a central bore, keyway, and set-screw holes.",
    "flange": "Flat disc with a bolt-hole circle and a raised sealing face.",
    "gear": "External spur/helical teeth around a central bore; web with lightening holes or a raised hub.",
    "nut": "Internally threaded hex prism, optionally flanged.",
    "pulley": "Grooved rim (V-belt or timing) around a bored hub.",
    "screw": "Threaded shank with a recessed drive head.",
    "shaft": "Turned cylinder with stepped diameters, keyways, and journals.",
}
DEFAULT_CHARACTERISTICS = "Machined mechanical component with bores and finished faces."

# --- Realistic PLM messiness -------------------------------------------------
# Real metadata stores rarely hold one clean spelling per grade. The same
# material shows up as a trade name, a DIN/EN number, an ISO designation, or a
# terse shop code — and some records carry nothing but a placeholder. We model
# that here and hand the predictor a normalizer so it can still converge.
MATERIAL_ALIASES: dict[str, list[str]] = {
    "Mild Steel (S235JR)": ["S235JR", "1.0038", "St37-2"],
    "Carbon Steel (C35)": ["C35", "1.0501"],
    "Carbon Steel (C45)": ["C45", "1.0503", "AISI 1045"],
    "Alloy Steel (42CrMo4)": ["42CrMo4", "1.7225", "AISI 4140"],
    "Case-Hardening Steel (16MnCr5)": ["16MnCr5", "1.7131"],
    "Case-Hardening Steel (20MnCr5)": ["20MnCr5", "1.7147", "20 MnCr 5"],
    "Case-Hardening Steel (18CrNiMo7-6)": ["18CrNiMo7-6", "1.6587"],
    "Bearing Steel (100Cr6)": ["100Cr6", "1.3505"],
    "Chrome Steel (AISI 52100)": ["52100", "1.3505"],
    "Stainless Steel (AISI 304)": ["304", "1.4301", "X5CrNi18-10"],
    "Stainless Steel (AISI 316)": ["316", "1.4401"],
    "Stainless Steel (AISI 440C)": ["440C", "1.4125"],
    "Stainless Steel (A2-70)": ["A2-70", "304 (A2)"],
    "Aluminum (6061-T6)": ["6061-T6", "AA6061", "3.3211"],
    "Aluminum (7075-T6)": ["7075-T6", "3.4365"],
    "Gray Cast Iron (EN-GJL-250)": ["EN-GJL-250", "GG25"],
    "Ductile Iron (EN-GJS-500)": ["EN-GJS-500", "GGG50"],
    "Property Class 8.8 Steel": ["Class 8.8", "Grade 8.8"],
    "Property Class 10.9 Steel": ["Class 10.9", "Grade 10.9"],
    "Property Class 12.9 Steel": ["Class 12.9", "Grade 12.9"],
}
JUNK_MATERIAL_TOKENS: tuple[str, ...] = ("TBD", "SEE DRAWING", "N/A", "-")
_JUNK_SET = {token.casefold() for token in JUNK_MATERIAL_TOKENS}

# Reverse lookup for normalization: any raw form (casefolded, and a
# space-stripped variant) -> canonical grade name.
_MATERIAL_ALIAS_LOOKUP: dict[str, str] = {}
for _canon, _aliases in MATERIAL_ALIASES.items():
    for _form in (_canon, *_aliases):
        _MATERIAL_ALIAS_LOOKUP[_form.casefold()] = _canon
        _MATERIAL_ALIAS_LOOKUP[_form.casefold().replace(" ", "")] = _canon

# Image-informed feature/teeth counts for the parts that appear in the demo.
# These are best-effort visual estimates (teeth for gears, holes/slots for
# brackets, grooves for pulleys) so the cost figures track the geometry you can
# actually see. Parts not listed fall back to a family-typical synthetic count.
CURATED_FEATURES: dict[str, int] = {
    # Example 1 — spur-gear query and its neighbours
    "gear/1": 20, "gear/99": 18, "gear/891": 8, "gear/780": 40, "gear/779": 34,
    "gear/39": 44, "gear/10": 12, "gear/883": 30, "gear/377": 12, "gear/658": 16,
    "gear/129": 22, "pulley/783": 4, "pulley/675": 24, "pulley/661": 26,
    # Example 2 — bevel-gear query and its neighbours
    "gear/1007": 16, "gear/157": 28, "gear/36": 24, "gear/313": 26, "gear/591": 30,
    "gear/105": 20, "gear/350": 30, "gear/109": 22, "gear/635": 24, "gear/619": 42,
    "gear/634": 18, "gear/412": 24, "gear/887": 20, "gear/542": 26,
    # Example 3 — sheet-metal bracket query and its neighbours
    "bracket/133": 6, "bracket/783": 6, "bracket/735": 8, "bracket/492": 4,
    "bracket/737": 6, "bracket/722": 9, "bracket/48": 6, "bracket/773": 8,
    "bracket/624": 5, "bracket/739": 6, "bracket/812": 8, "bracket/738": 5,
    "bracket/744": 4, "bracket/731": 6,
}


def _hash(name: str, salt: str) -> int:
    return int(hashlib.md5(f"{salt}:{name}".encode("utf-8")).hexdigest()[:8], 16)


def _category_of(path: str) -> str:
    """Return the part family (parent folder) for a CAD id, lowercased.

    Returns an empty string when the id has no recognized family folder.
    """
    parent = Path(path).parent.name.strip().lower()
    return parent if parent in CATEGORY_PROFILES else ""


def _weighted_pick(options: Sequence[tuple[str, float]], stem: str, salt: str) -> str:
    """Deterministically pick one weighted option from the family's list."""
    total = sum(weight for _, weight in options)
    target = (_hash(stem, salt) % 1000) / 1000.0 * total
    cumulative = 0.0
    for name, weight in options:
        cumulative += weight
        if target < cumulative:
            return name
    return options[-1][0]


def compute_true_cost_from_equation(material: str, process: str, internal_features: int) -> float:
    base = MATERIAL_COST_INDEX[material] ** 1.5 / 10
    feat = 1 + (0.08 * internal_features) ** 2.2
    return base * feat * PROCESS_MULTIPLIER[process]


def metadata_for(path: str) -> dict:
    """Return deterministic, family-aware synthetic metadata for one CAD id."""
    category = _category_of(path)
    profile = CATEGORY_PROFILES.get(category, DEFAULT_PROFILE)
    stem = Path(path).stem
    material = _weighted_pick(profile["materials"], stem, "mat")
    process = _weighted_pick(profile["processes"], stem, "proc")
    # Not every family is heat treated (a sheet-metal bracket or a cast pulley
    # typically is not), so this stays None rather than inventing a step.
    heat_treatments = profile.get("heat_treatments")
    heat_treatment = _weighted_pick(heat_treatments, stem, "heat") if heat_treatments else None
    curated_key = f"{category}/{stem}" if category else stem
    features = CURATED_FEATURES.get(
        curated_key, profile["feature_base"] + _hash(stem, "feat") % profile["feature_span"]
    )
    noise = (_hash(stem, "noise") % 200) / 1000.0 - 0.10   # +/- 10 %
    cost = round(compute_true_cost_from_equation(material, process, features) * (1 + noise), 2)
    return {
        "PartFamily": category.capitalize() if category else "Machined Part",
        "Characteristics": CATEGORY_CHARACTERISTICS.get(category, DEFAULT_CHARACTERISTICS),
        "Material": material,
        "Process": process,
        "HeatTreatment": heat_treatment,
        "ProcessRoute": profile["route"],
        "InternalFeatures": features,
        "Cost": cost,
    }


def canonical_material(value) -> str | None:
    """Normalize a messy/legacy material string to its canonical grade.

    Trade names, DIN/EN numbers and shop codes (e.g. "1.7147", "20 MnCr 5")
    collapse onto the same canonical grade; placeholder junk ("TBD", "N/A", ...)
    maps to None so it is ignored during aggregation. Pass this as the
    ``normalize`` callable on the Material CategoricalRule.
    """
    if value is None:
        return None
    text = " ".join(str(value).split())
    low = text.casefold()
    if low in _JUNK_SET:
        return None
    return _MATERIAL_ALIAS_LOOKUP.get(low) or _MATERIAL_ALIAS_LOOKUP.get(low.replace(" ", "")) or text


def short_id(path: str) -> str:
    """Return "<family>/<filename>" for display, dropping the long absolute path.

    Use this only for what you print/show a human (e.g. in ``hits_table``); the
    Context Layer itself keeps using the full id so it can look up metadata.
    """
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def display_hits(hits, metadata_by_id: Mapping[str, dict]):
    """Return a ``(hits, metadata_by_id)`` pair with ids shortened to
    ``short_id`` for compact display in ``hits_table``.

    The original ``hits`` (full-path ids) still needs to be passed to
    ``predictor.infer`` and friends — only use this shortened copy for display.
    """
    from types import SimpleNamespace

    short_hits = [SimpleNamespace(id=short_id(h.id), score=h.score) for h in hits]
    short_metadata = {short_id(pid): meta for pid, meta in metadata_by_id.items()}
    return short_hits, short_metadata


def stored_family(path: str) -> str:
    """Return the family label the source dataset stores for a part (its
    folder), independent of what the neighbours suggest.

    Comparing this against a predicted ``PartFamily`` is how you flag a
    mislabeled record in the store.
    """
    category = _category_of(path)
    return category.capitalize() if category else "Unknown"


def expert_summary(category: str) -> dict[str, str]:
    """Return the process route and expert note for a part family.

    Handy for a "what an expert would say about this part" view once the
    Context Layer has predicted the family or process route.
    """
    key = category.strip().lower()
    profile = CATEGORY_PROFILES.get(key, DEFAULT_PROFILE)
    return {
        "family": key or "machined part",
        "process_route": profile["route"],
        "note": EXPERT_NOTES.get(key, EXPERT_NOTES["_default"]),
    }


def _iter_ids(items):
    """Yield CAD-file ids from a mixed iterable.

    Accepts: VectorHit, plain strings/paths, or nested iterables of either
    (so ``search_by_shape`` results work whether they're flat or batched).
    """
    for item in items:
        if hasattr(item, "id"):
            yield item.id
        elif isinstance(item, (str, Path)):
            yield str(item)
        else:
            yield from _iter_ids(item)


def build_records(hits_or_paths) -> dict[str, dict]:
    """Build a ``{path: metadata}`` dict from VectorHits or CAD file paths."""
    
    return {pid: metadata_for(pid) for pid in _iter_ids(hits_or_paths)}


class SyntheticPLM:
    """Stand-in for an external PLM / ERP system.

    This class plays the role your production metadata store plays: given a list
    of part ids it returns whatever records it holds, and it accepts writes. In
    your own project you would NOT reimplement this — you would call your
    existing PLM/ERP client instead. It exists here only so the tutorial runs
    without a real backend.

    Records are family-aware and hash-derived from the file stem, so the same id
    always returns the same metadata across runs and machines, and parts in the
    same family (gear, shaft, ...) share a coherent material palette, primary
    process, and process route. To stay realistic, the backend deliberately
    leaves some records incomplete — like a PLM where many parts are not yet
    fully tagged:

    * **The query itself (and any identical indexed twin) is returned empty.**
      The demo query files are copies of indexed parts, so the top hit and its
      same-stem duplicate are stripped — otherwise the predictor would just copy
      the query's own record instead of aggregating from genuine neighbours.
    * **The remaining ids follow a 50 / 30 / 20 mix:** ~50 % full records,
      ~30 % only the textual tags (family, characteristics, material, process,
      heat treatment, route) with the numeric estimates pending, ~20 % empty.

    Because neighbours of a shape query belong to the same family, agreement on
    the process route and primary process arises naturally from the data — no
    artificial consensus is injected. Explicit writes via :meth:`store` always
    win over the synthetic record.
    """

    NUMERIC_FIELDS: tuple[str, ...] = ("Cost",)
    TEXTUAL_FIELDS: tuple[str, ...] = (
        "PartFamily", "Characteristics", "Material", "Process", "HeatTreatment", "ProcessRoute",
    )

    _FULL_BUCKET_CEILING = 6      # 0-5 → ~50 % full
    _PARTIAL_BUCKET_CEILING = 8   # 6-7 → ~30 % partial (textual tags only)
    # 8-9 → ~20 % empty

    def __init__(self) -> None:
        self._written: dict[str, dict] = {}

    def fetch(self, part_ids: Sequence[str]) -> dict[str, dict]:
        """Return ``{part_id: record}`` for the requested ids, in order.

        This is the read endpoint. Your real backend would issue one batched
        query here and return the rows it finds (omitting unknown ids).
        """
        ids_list = list(part_ids)
        # The query file is a copy of an indexed part, so its identical twin
        # (same file stem) would otherwise leak the answer. Strip both.
        query_stem = Path(ids_list[0]).stem if ids_list else None
        out: dict[str, dict] = {}
        for index, pid in enumerate(ids_list):
            if index == 0 or (query_stem is not None and Path(pid).stem == query_stem):
                # Strip the query and its exact duplicate so the predictor has
                # to aggregate from genuine neighbours. Overrides still win.
                record: dict = {}
            else:
                record = self._shape_for(pid)
            overrides = self._written.get(pid)
            if overrides:
                record = {**record, **overrides}
            out[pid] = record
        return out

    def store(self, updates: Mapping[str, dict]) -> None:
        """Persist explicit per-part overrides (the write endpoint)."""
        for pid, payload in updates.items():
            self._written.setdefault(str(pid), {}).update(payload)

    # --- synthetic data generation (your real backend already owns the data) ---

    @staticmethod
    def _missingness_bucket(pid: str) -> int:
        """Stable 0\u20139 bucket per file stem; controls metadata completeness."""
        return _hash(Path(pid).stem, "missing") % 10

    @classmethod
    def _shape_for(cls, pid: str) -> dict:
        """Return the synthetic record for a part after applying the
        per-file missingness rule. Always returns a dict (possibly empty).
        """
        bucket = cls._missingness_bucket(pid)
        if bucket >= cls._PARTIAL_BUCKET_CEILING:
            return {}
        record = metadata_for(pid)
        # Model messy material tagging: mostly the clean grade, sometimes a
        # legacy/DIN code, occasionally placeholder junk. Cost is unaffected
        # (it was derived from the true grade) — only the tag text varies.
        if "Material" in record:
            record = {**record, "Material": cls._messy_material(pid, record["Material"])}
        if bucket >= cls._FULL_BUCKET_CEILING:
            # Partial: keep only the textual tags a human is likely to have
            # entered; drop the numeric estimates (Cost) and feature count.
            return {field: record[field] for field in cls.TEXTUAL_FIELDS}
        return record

    @staticmethod
    def _messy_material(pid: str, canonical: str) -> str:
        """Return the material as a real PLM might store it: mostly canonical,
        sometimes a legacy/DIN code, occasionally placeholder junk.
        """
        stem = Path(pid).stem
        bucket = _hash(stem, "matmess") % 10
        if bucket >= 9:  # ~10% placeholder junk
            return JUNK_MATERIAL_TOKENS[_hash(stem, "junk") % len(JUNK_MATERIAL_TOKENS)]
        if bucket >= 6:  # ~30% legacy/alias code
            aliases = MATERIAL_ALIASES.get(canonical)
            if aliases:
                return aliases[_hash(stem, "alias") % len(aliases)]
        return canonical


class OnDemandContextProvider(ContextProvider):
    """Adapter exposing a PLM/ERP backend to the HOOPS AI Context Layer.

    This is the only class you implement for your own system. The Context Layer
    calls exactly three methods on a ``ContextProvider``; each one here is a
    thin translation to the backend:

    * :meth:`get_contexts`     -> ``backend.fetch``  (read metadata by part id)
    * :meth:`set_contexts`     -> ``backend.store``  (write metadata back)
    * :meth:`list_numeric_keys` -> which keys are numeric (so numeric
      aggregation rules fire instead of categorical ones)

    In production, swap ``SyntheticPLM`` for your real PLM/ERP client and
    forward these three calls to its read / write / schema endpoints — the rest
    of the Context Layer is unchanged.
    """

    def __init__(
        self,
        backend: SyntheticPLM | None = None,
        numeric_keys: Sequence[str] | None = None,
    ) -> None:
        self._backend = backend if backend is not None else SyntheticPLM()
        self._numeric_keys = (
            tuple(numeric_keys)
            if numeric_keys is not None
            else tuple(self._backend.NUMERIC_FIELDS)
        )

    def get_contexts(self, part_ids: Sequence[str]) -> Mapping[str, dict]:
        return self._backend.fetch(list(_iter_ids(part_ids)))

    def set_contexts(self, updates: Mapping[str, dict]) -> None:
        self._backend.store(updates)

    def list_numeric_keys(self) -> Sequence[str]:
        return self._numeric_keys
