"""
PAVE-ANN v2.0 . batch scoring.

    python cli.py --input designs.csv --output predictions.csv
    python cli.py --template > blank_design.csv
"""
import argparse
import sys
from pathlib import Path

from predictor import PaveANNPredictor


def main() -> int:
    ap = argparse.ArgumentParser(description="PAVE-ANN batch inference")
    ap.add_argument("--model-dir", default=str(Path(__file__).parent))
    ap.add_argument("--input", "-i")
    ap.add_argument("--output", "-o", default="pave_ann_predictions.csv")
    ap.add_argument("--template", action="store_true",
                    help="emit a valid median design to stdout and exit")
    ap.add_argument("--design-life", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    P = PaveANNPredictor.load(a.model_dir, device=a.device)

    if a.template:
        P.template().to_csv(sys.stdout, index=False)
        return 0
    if not a.input:
        ap.error("--input is required unless --template is given")

    import pandas as pd
    df = pd.read_csv(a.input)
    out = P.predict(df)
    if a.design_life:
        out = pd.concat([out, P.design_life(out, df["E_ACB"])], axis=1)
    pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)],
              axis=1).to_csv(a.output, index=False)

    n_out = int((out["n_params_out_of_range"] > 0).sum())
    print(f"scored {len(df)} designs -> {a.output}")
    if n_out:
        print(f"warning: {n_out} designs fall outside the training envelope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
