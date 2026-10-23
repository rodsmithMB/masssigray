"""Energy grids and dwell times for an XAS scan."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .xray import E_of_k, k_of_E


@dataclass
class ScanPlan:
    """Three-region scan: pre-edge, XANES (energy steps), EXAFS (k steps).

    Energies are relative to E0 in eV. EXAFS dwell grows as
    dwell_exafs * (k / k_start)**k_time_weight.
    i0_dwell_factor sets the I0 (blank) scan dwell relative to the sample scan.
    """

    E0: float
    pre_start: float = -150.0
    pre_step: float = 5.0
    xanes_start: float = -30.0
    xanes_end: float = 40.0
    xanes_step: float = 0.5
    kmax: float = 12.0
    kstep: float = 0.05
    dwell_pre: float = 1.0
    dwell_xanes: float = 1.0
    dwell_exafs: float = 1.0
    k_time_weight: float = 0.0
    i0_dwell_factor: float = 1.0

    def grid(self):
        """Return (energy, dwell) arrays."""
        pre = np.arange(self.pre_start, self.xanes_start, self.pre_step)
        xan = np.arange(self.xanes_start, self.xanes_end, self.xanes_step)
        k0 = float(k_of_E(self.E0 + self.xanes_end, self.E0))
        ks = np.arange(k0, self.kmax + self.kstep / 2, self.kstep) if self.kmax > k0 else np.array([])
        E = np.concatenate([self.E0 + pre, self.E0 + xan, E_of_k(ks, self.E0)])
        kw = (ks / k0) ** self.k_time_weight if ks.size else ks
        t = np.concatenate([np.full(pre.size, self.dwell_pre),
                            np.full(xan.size, self.dwell_xanes),
                            self.dwell_exafs * kw])
        return E, t

    @property
    def emin(self):
        return self.E0 + self.pre_start

    @property
    def emax(self):
        return float(self.grid()[0][-1])

    def total_time(self) -> tuple[float, float]:
        """(sample scan, I0 scan) counting time in s, excluding motor overhead."""
        _, t = self.grid()
        return float(t.sum()), float(t.sum() * self.i0_dwell_factor)
