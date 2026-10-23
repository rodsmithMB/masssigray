"""Materials: chemical formulas, weight-percent mixtures, densities, presets.

Every material is reduced to a dict of element mass fractions, which is all the
absorption calculations need. A material can be a single formula
("Fe2O3", "Ni0.5Fe0.5(OH)2") or a weight-percent mixture of formulas
({"Pt": 5, "Al2O3": 95}).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import xraydb

# Matrix presets: (formula, nominal bulk density in g/cm3).
# Densities are crystal/solid values; pressed pellets are lower (see packing
# fraction in pellet.py).
MATRIX_PRESETS = {
    "Boron nitride (BN)": ("BN", 2.10),
    "Cellulose": ("C6H10O5", 1.50),
    "Polyethylene (PE)": ("C2H4", 0.94),
    "Graphite": ("C", 2.20),
    "Sucrose": ("C12H22O11", 1.59),
    "PVP": ("C6H9NO", 1.20),
    "Starch": ("C6H10O5", 1.50),
    "Lithium carbonate": ("Li2CO3", 2.11),
    "Silica (SiO2)": ("SiO2", 2.20),
    "Alumina (Al2O3)": ("Al2O3", 3.95),
}


def parse_formula(formula: str) -> dict[str, float]:
    """Return {element: moles} for a formula; parentheses and decimals allowed."""
    formula = formula.strip()
    if not formula:
        raise ValueError("empty formula")
    try:
        comp = xraydb.chemparse(formula)
    except Exception as err:  # xraydb raises a generic Warning/ValueError
        raise ValueError(f"cannot parse formula '{formula}': {err}") from err
    if not comp:
        raise ValueError(f"cannot parse formula '{formula}'")
    return {el: float(n) for el, n in comp.items() if n > 0}


def molar_mass(formula: str) -> float:
    """Molar mass in g/mol."""
    return sum(n * xraydb.atomic_mass(el) for el, n in parse_formula(formula).items())


def formula_mass_fractions(formula: str) -> dict[str, float]:
    comp = parse_formula(formula)
    masses = {el: n * xraydb.atomic_mass(el) for el, n in comp.items()}
    total = sum(masses.values())
    return {el: m / total for el, m in masses.items()}


def parse_mixture(text: str) -> dict[str, float]:
    """Parse "Pt:5, Al2O3:95" (wt%) into {formula: wt%}.

    A single component without a weight ("Fe2O3") gets 100 wt%. One component
    may omit its weight and takes the balance: "Pt:5, Al2O3".
    """
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("empty composition")
    comps: dict[str, float | None] = {}
    for p in parts:
        if ":" in p:
            f, w = p.split(":", 1)
            comps[f.strip()] = float(w.strip().rstrip("%"))
        else:
            comps[p] = None
    missing = [f for f, w in comps.items() if w is None]
    if len(missing) > 1:
        raise ValueError("only one component may omit its wt% (the balance)")
    given = sum(w for w in comps.values() if w is not None)
    if missing:
        if given > 100:
            raise ValueError("weights exceed 100 wt%")
        comps[missing[0]] = 100.0 - given
    return {f: float(w) for f, w in comps.items()}


@dataclass
class Material:
    """A material defined by wt% of formula components, with optional density.

    density is in g/cm3 and may be None when unknown (thickness-based results
    are then unavailable).
    """

    components: dict[str, float]
    density: float | None = None
    name: str = ""
    _wfrac: dict[str, float] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.components:
            raise ValueError("material has no components")
        total = sum(self.components.values())
        if total <= 0:
            raise ValueError("component weights must be positive")
        wf: dict[str, float] = {}
        for formula, w in self.components.items():
            for el, f in formula_mass_fractions(formula).items():
                wf[el] = wf.get(el, 0.0) + f * w / total
        self._wfrac = wf
        if not self.name:
            self.name = self.label

    @classmethod
    def from_formula(cls, formula: str, density: float | None = None, name: str = ""):
        return cls({formula: 100.0}, density=density, name=name)

    @classmethod
    def from_text(cls, text: str, density: float | None = None, name: str = ""):
        """Formula or wt% mixture text, e.g. "CuO" or "Pt:5, Al2O3"."""
        return cls(parse_mixture(text), density=density, name=name)

    @property
    def label(self) -> str:
        if len(self.components) == 1:
            return next(iter(self.components))
        return " + ".join(f"{w:g}% {f}" for f, w in self.components.items())

    @property
    def mass_fractions(self) -> dict[str, float]:
        return dict(self._wfrac)

    def mass_fraction(self, element: str) -> float:
        return self._wfrac.get(element, 0.0)

    @property
    def elements(self) -> list[str]:
        return sorted(self._wfrac, key=xraydb.atomic_number)


def matrix_preset(name: str) -> Material:
    formula, rho = MATRIX_PRESETS[name]
    return Material.from_formula(formula, density=rho, name=name)
