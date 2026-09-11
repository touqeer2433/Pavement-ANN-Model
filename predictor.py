"""
PAVE-ANN v2.0 . standalone inference API.

Depends only on torch, numpy and pandas. The TorchScript graph already contains
feature expansion, normalisation, the mechanistic term, the ensemble and the
inverse target transform, so there is no preprocessing to reproduce here.

    from predictor import PaveANNPredictor
    p = PaveANNPredictor.load("./")
    out = p.predict_csv("designs.csv")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch


class InputValidationError(ValueError):
    pass


class PaveANNPredictor:
    def __init__(self, model: torch.jit.ScriptModule, meta: Dict, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.meta = meta
        self.device = device
        self.features: List[str] = meta["inputs"]["order"]
        self.targets: List[str] = meta["outputs"]["order"]
        self.primary: List[str] = meta["outputs"]["primary"]
        self.units: Dict[str, str] = meta["outputs"]["units"]
        self.lo = np.array([meta["inputs"]["training_min"][f] for f in self.features])
        self.hi = np.array([meta["inputs"]["training_max"][f] for f in self.features])
        self.median = np.array([meta["inputs"]["training_median"][f] for f in self.features])

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, folder: Union[str, Path], device: str = "cpu") -> "PaveANNPredictor":
        folder = Path(folder)
        meta = json.loads((folder / "metadata.json").read_text())
        model = torch.jit.load(str(folder / "pave_ann.ts"), map_location=device)
        return cls(model, meta, device)

    # -------------------------------------------------------------- validate
    def _frame_to_array(self, X: Union[pd.DataFrame, np.ndarray, Dict]) -> np.ndarray:
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.features if f not in X.columns]
            if missing:
                raise InputValidationError(
                    f"{len(missing)} required parameters are missing: {missing[:8]}"
                    + (" ..." if len(missing) > 8 else ""))
            A = X[self.features].to_numpy(float)
        else:
            A = np.asarray(X, float)
            if A.ndim == 1:
                A = A[None, :]
            if A.shape[1] != len(self.features):
                raise InputValidationError(
                    f"expected {len(self.features)} columns in the documented order, "
                    f"got {A.shape[1]}")
        if not np.isfinite(A).all():
            raise InputValidationError("input contains NaN or infinite values")
        return A

    def range_check(self, A: np.ndarray) -> pd.DataFrame:
        """Per-row report of how far outside the training envelope each design sits."""
        below = np.maximum(self.lo - A, 0) / (self.hi - self.lo + 1e-12)
        above = np.maximum(A - self.hi, 0) / (self.hi - self.lo + 1e-12)
        ex = below + above
        worst = ex.argmax(1)
        return pd.DataFrame({
            "n_params_out_of_range": (ex > 0).sum(1),
            "max_extrapolation_frac": ex.max(1),
            "worst_parameter": [self.features[i] for i in worst],
        })

    # --------------------------------------------------------------- predict
    def predict(self, X, batch_size: int = 4096, with_uncertainty: bool = True
                ) -> pd.DataFrame:
        A = self._frame_to_array(X)
        chunks = []
        with torch.no_grad():
            for s in range(0, len(A), batch_size):
                t = torch.tensor(A[s:s + batch_size], dtype=torch.float32,
                                 device=self.device)
                mean, std, alea, epi = self.model(t)
                chunks.append([mean.cpu().numpy(), std.cpu().numpy(),
                               alea.cpu().numpy(), epi.cpu().numpy()])
        mean = np.vstack([c[0] for c in chunks])
        std = np.vstack([c[1] for c in chunks])
        alea = np.vstack([c[2] for c in chunks])
        epi = np.vstack([c[3] for c in chunks])

        out = pd.DataFrame(mean, columns=self.targets)
        if with_uncertainty:
            for j, t in enumerate(self.targets):
                out[f"{t}_sd"] = std[:, j]
                out[f"{t}_lo95"] = mean[:, j] - 1.96 * std[:, j]
                out[f"{t}_hi95"] = mean[:, j] + 1.96 * std[:, j]
            for j, t in enumerate(self.primary):
                k = self.targets.index(t)
                out[f"{t}_epistemic_sd"] = epi[:, k]
                out[f"{t}_aleatoric_sd"] = alea[:, k]
        rc = self.range_check(A)
        for c in rc.columns:
            out[c] = rc[c].to_numpy()
        if isinstance(X, pd.DataFrame):
            out.index = X.index
        return out

    def predict_csv(self, path: Union[str, Path],
                    output: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        df = pd.read_csv(path)
        res = self.predict(df)
        if output:
            pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)],
                      axis=1).to_csv(output, index=False)
        return res

    def template(self, n: int = 1) -> pd.DataFrame:
        """A valid starting design at the training median."""
        return pd.DataFrame(np.tile(self.median, (n, 1)), columns=self.features)

    # ---------------------------------------------------- design transfer fns
    def design_life(self, pred: pd.DataFrame, E_ac_MPa: Union[float, np.ndarray],
                    fatigue_coeffs=(0.0796, 3.291, 0.854),
                    rutting_coeffs=(1.365e-9, 4.477)) -> pd.DataFrame:
        """
        Allowable repetitions from the Asphalt Institute transfer functions.

            Nf = k1 * eps_t^(-k2) * E1^(-k3)      E1 in psi, eps_t in strain
            Nd = c1 * eps_c^(-c2)

        Coefficients are the published defaults and are meant to be replaced by
        locally calibrated values. The strain channels are converted from the
        microstrain units the network emits.
        """
        k1, k2, k3 = fatigue_coeffs
        c1, c2 = rutting_coeffs
        eps_t = np.abs(pred["eps_t_ACB"].to_numpy()) * 1e-6
        eps_c = np.abs(pred["eps_c_SG"].to_numpy()) * 1e-6
        E_psi = np.asarray(E_ac_MPa, float) * 145.0377
        with np.errstate(divide="ignore", invalid="ignore"):
            Nf = k1 * eps_t ** (-k2) * E_psi ** (-k3)
            Nd = c1 * eps_c ** (-c2)
        return pd.DataFrame({
            "N_fatigue_allowable": Nf,
            "N_rutting_allowable": Nd,
            "governing_mode": np.where(Nf <= Nd, "fatigue", "subgrade rutting"),
            "N_design": np.minimum(Nf, Nd),
        }, index=pred.index)


if __name__ == "__main__":
    p = PaveANNPredictor.load(Path(__file__).parent)
    demo = p.template()
    print(p.predict(demo).T)
