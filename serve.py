"""
PAVE-ANN v2.0 . FastAPI inference service.

    uvicorn serve:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health          liveness
    GET  /metadata        input order, units, valid ranges, test metrics
    GET  /template        a valid design at the training median
    POST /predict         {"designs": [{param: value, ...}, ...]}
    POST /predict_csv     multipart CSV upload, returns CSV
"""
import io
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from predictor import InputValidationError, PaveANNPredictor

HERE = Path(__file__).parent
PRED = PaveANNPredictor.load(HERE)

app = FastAPI(title="PAVE-ANN", version="2.0",
              description="Neural network surrogate for layered pavement response")


class PredictRequest(BaseModel):
    designs: List[Dict[str, float]] = Field(..., description="one dict per design")
    include_design_life: bool = False
    E_ac_MPa: Optional[float] = Field(None, description="AC modulus for the transfer functions")


@app.get("/health")
def health():
    return {"status": "ok", "model": PRED.meta["name"],
            "version": PRED.meta["version"],
            "fingerprint": PRED.meta["fingerprint"],
            "ensemble_members": PRED.meta["n_ensemble_members"]}


@app.get("/metadata")
def metadata():
    return PRED.meta


@app.get("/template")
def template():
    return PRED.template().to_dict("records")[0]


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame(req.designs)
        out = PRED.predict(df)
        if req.include_design_life:
            E = req.E_ac_MPa if req.E_ac_MPa is not None else df.get("E_ACB")
            if E is None:
                raise InputValidationError("E_ac_MPa or an E_ACB column is required")
            out = pd.concat([out, PRED.design_life(out, E)], axis=1)
        return {"n": len(out), "predictions": out.to_dict("records")}
    except InputValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        out = pd.concat([df.reset_index(drop=True),
                         PRED.predict(df).reset_index(drop=True)], axis=1)
    except InputValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    buf = io.StringIO(); out.to_csv(buf, index=False); buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition":
                                      "attachment; filename=pave_ann_predictions.csv"})
