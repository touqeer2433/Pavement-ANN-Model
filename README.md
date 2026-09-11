# PAVE-ANN v2.0 . deployment bundle

Deep neural network surrogate for layered flexible pavement response.
Fingerprint `e1340a664004` . 5 ensemble members
. generated 2026-09-11.

## Inputs
43 raw design parameters, exact order and units in
`metadata.json`. Feature expansion and scaling live inside the traced graph,
so pass raw physical values.

## Outputs
9 channels, 4 primary:

- `eps_t_ACB` . Fatigue tensile strain, AC base [microstrain]
- `eps_c_SG` . Rutting compressive strain, subgrade [microstrain]
- `U2_surface` . Surface deflection [mm]
- `sigma_vm_max` . Peak von Mises stress [MPa]

Each comes with an aleatoric standard deviation, an epistemic standard
deviation from ensemble disagreement, and a 95% interval.

## Test-set accuracy
- `eps_t_ACB`: R2 0.89731, MAPE 8.42%
- `eps_c_SG`: R2 0.97784, MAPE 2.82%
- `U2_surface`: R2 0.97263, MAPE 3.00%
- `sigma_vm_max`: R2 0.96279, MAPE 3.69%

## Quick start
```bash
pip install -r requirements.txt
python cli.py --template > design.csv     # edit design.csv
python cli.py -i design.csv -o out.csv --design-life
```

```python
from predictor import PaveANNPredictor
p = PaveANNPredictor.load(".")
out = p.predict(p.template())
```

## Service
```bash
uvicorn serve:app --port 8000
curl localhost:8000/health
```

## Interactive
```bash
streamlit run app_streamlit.py
```

## Docker
```bash
docker build -t pave-ann:2.0 . && docker run -p 8000:8000 pave-ann:2.0
```

## Limits
Predictions are trustworthy inside the training envelope recorded in
`metadata.json`. `predict()` returns `n_params_out_of_range` and
`max_extrapolation_frac` per row: treat any row with a non-zero count as
indicative and verify with FEM. The Asphalt Institute transfer functions in
`design_life()` carry published default coefficients and should be replaced
with locally calibrated ones before use in design.
