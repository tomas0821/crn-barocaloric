"""Reusable quasi-direct barocaloric core (used by step 3 Monte Carlo and step 4 blind validation).

The physics matches step2_quasidirect.py, generalized to either sign of dT_t/dP. Units are per kg
throughout.

Given:
  T [K] grid; excess heat capacity of the transition Cex(T) [J/kg/K], measured on WARMING;
  background Cbg(T) [J/kg/K]; signed slope a = dT_t/dP [K/GPa]; hysteresis h [K] (T_heat - T_cool > 0);
  specific volume v [m^3/kg]; volumetric thermal expansion alphaV(T) [1/K].
Entropy branches:
  S_ex(T): cumulative ∫ Cex/T dT (0 in the low-T phase, ΔS_t in the high-T phase)
  warming branch at P: S_ex(T - aP);  cooling branch at P: S_ex(T - aP + h)
  lattice: -v alphaV P (always reversible)
Reversible cycle:
  a > 0 (conventional; compression drives high-T -> low-T, so it heats):
      ΔS_rev,tr = min(0, S_ex(T - aP + h) - S_ex(T))       [cooling@P vs heating@0]
  a < 0 (inverse; compression drives low-T -> high-T, so it cools):
      ΔS_rev,tr = max(0, S_ex(T - aP) - S_ex(T + h))       [heating@P vs cooling@0]
  Adiabatic ΔT_rev: start from the appropriate branch at P = 0. The transformed fraction can only
  advance in the direction the pressure drives it, so take min (a>0) or max (a<0) of the two branch
  entropies. Solve isentropically by bisection.
"""
import numpy as np


def cumint(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def gaussian_broaden(T, y, sigma):
    """Convolve y(T) on a uniform grid with a normalized Gaussian of std sigma [K]."""
    if sigma <= 1e-6:
        return y
    dT = T[1] - T[0]
    n = int(np.ceil(5 * sigma / dT))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * dT / sigma) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


class Baro:
    def __init__(self, T, Cex, Cbg, a, h, v, alphaV, dS_target=None, sigma_broad=0.0, a_heat=None, a_cool=None,
                 b=0.0, c=0.0):
        """T: uniform grid [K]; Cex, Cbg: arrays on T [J/kg/K]; alphaV: callable or float.
        a_heat / a_cool: optional branch-specific initial slopes [K/GPa] (default: both = a).
        b: curvature of T_t(P) [K/GPa^2]; the shift is a*P + b*P^2.
        c: relative decline of ΔS_t per GPa; S_ex is scaled by (1 - c*P) (sensitivity; default 0)."""
        self.T = np.asarray(T, float)
        cex = gaussian_broaden(self.T, np.asarray(Cex, float), sigma_broad)
        sex = cumint(cex / self.T, self.T)
        if dS_target is not None:
            sex *= dS_target / sex[-1]
        self.Sex_tab = sex
        self.dS = sex[-1]
        self.Sbg_tab = cumint(np.asarray(Cbg, float) / self.T, self.T)
        self.a, self.h, self.v = float(a), float(h), float(v)
        self.a_h = float(a if a_heat is None else a_heat)
        self.a_c = float(a if a_cool is None else a_cool)
        self.b, self.c = float(b), float(c)
        self.alphaV = alphaV if callable(alphaV) else (lambda T, x=float(alphaV): x + 0 * T)

    def S_branch(self, T, P, branch):
        """Transition entropy on the warming ('heat') or cooling ('cool') branch at pressure P."""
        if branch == "heat":
            return self.k(P) * self.S_ex(T - self.sh_h(P))
        return self.k(P) * self.S_ex(T - self.sh_c(P) + self.h)

    def dS_branch(self, T, P, branch):
        """Single-branch (isobaric-pair) transition ΔS = S_branch(T, P) - S_branch(T, 0)."""
        return self.S_branch(T, P, branch) - self.S_branch(T, 0.0, branch)

    def dS_rev_overlap(self, T, P):
        """Overlap construction (e.g. Stern-Taulats 2014 Fig. 7): min(|ΔS_cool|, |ΔS_heat|), signed."""
        dc, dh = self.dS_branch(T, P, "cool"), self.dS_branch(T, P, "heat")
        m = np.minimum(np.abs(dc), np.abs(dh))
        return np.sign(self.a) * -m if self.a >= 0 else m

    def S_ex(self, T):
        return np.interp(T, self.T, self.Sex_tab, left=0.0, right=self.dS)

    def S_bg(self, T):
        return np.interp(T, self.T, self.Sbg_tab)

    def sh_h(self, P):
        return self.a_h * P + self.b * P * P

    def sh_c(self, P):
        return self.a_c * P + self.b * P * P

    def k(self, P):
        return 1.0 - self.c * P

    def lat(self, T, P):
        return -self.v * self.alphaV(T) * P * 1e9

    def dS_rev_trans(self, T, P):
        """Cross-branch construction (conservative), PHASE-CONVERSION part only: cooling@P vs heating@0 (a>0),
        heating@P vs cooling@0 (a<0). With c ≠ 0 the continuous part from the phase-dependent thermal
        expansion is returned separately by phase_lat()."""
        h = self.h
        if self.a >= 0:
            return self.k(P) * np.minimum(0.0, self.S_ex(T - self.sh_c(P) + h) - self.S_ex(T))
        return self.k(P) * np.maximum(0.0, self.S_ex(T - self.sh_h(P)) - self.S_ex(T + h))

    def phase_lat(self, T, P):
        """Continuous entropy change of the high-T-phase fraction when ΔS_t declines with P. Equivalent to an
        extra thermal expansion Δα = c·ΔS_t/v of the high-T phase. It is (k-1)·S_ex(initial state)."""
        s0 = self.S_ex(T) if self.a >= 0 else self.S_ex(T + self.h)
        return (self.k(P) - 1.0) * s0

    def dS_first(self, T, P):
        """Single-branch (first-application, upper-bound) transition ΔS: warming@P vs warming@0."""
        return self.k(P) * self.S_ex(T - self.sh_h(P)) - self.S_ex(T)

    def dT_rev(self, T, P, iters=40):
        h = self.h
        if self.a >= 0:
            s0 = self.S_ex(T); target = self.S_bg(T) + s0
            lo, hi = T - 3.0, T + 80.0
            comb = lambda Tp: self.k(P) * np.minimum(s0, self.S_ex(Tp - self.sh_c(P) + h))
        else:
            s0 = self.S_ex(T + h); target = self.S_bg(T) + s0
            lo, hi = T - 80.0, T + 3.0
            comb = lambda Tp: self.k(P) * np.maximum(s0, self.S_ex(Tp - self.sh_h(P)))
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            g = self.S_bg(mid) + self.lat(mid, P) + comb(mid) - target
            hi = np.where(g > 0, mid, hi); lo = np.where(g > 0, lo, mid)
        return 0.5 * (lo + hi) - T

    def summary(self, Tgrid, Pgrid, P_dT=(0.25, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0)):
        """Return f(P), peak |ΔS_rev| (incl. lattice) and peak |ΔS_first| on Pgrid, and peak |ΔT_rev| at P_dT."""
        f = np.empty(len(Pgrid)); dSr = np.empty(len(Pgrid)); dS1 = np.empty(len(Pgrid))
        for i, p in enumerate(Pgrid):
            tr = self.dS_rev_trans(Tgrid, p)
            f[i] = np.abs(tr).max() / self.dS
            dSr[i] = np.abs(tr + self.lat(Tgrid, p) + self.phase_lat(Tgrid, p)).max()
            dS1[i] = np.abs(self.dS_first(Tgrid, p)).max()
        dT = {p: np.abs(self.dT_rev(Tgrid, p)).max() for p in P_dT}
        return f, dSr, dS1, dT


def first_crossing(P, f, th):
    j = np.where(f >= th)[0]
    return P[j[0]] if j.size else np.nan
