import io
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import tritonclient.http.aio as triton
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "cifar_cnn")
STATS_PATH = os.environ.get("STATS_PATH", "serving_data/stats.json")

with open(STATS_PATH) as f:
    stats = json.load(f)
MEAN = np.array(stats["mean"], dtype=np.float32)
STD = np.array(stats["std"], dtype=np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = triton.InferenceServerClient(TRITON_URL)
    yield
    await app.state.client.close()


app = FastAPI(lifespan=lifespan)


def preprocess(content: bytes) -> Any:
    image = Image.open(io.BytesIO(content)).convert("RGB").resize((32, 32))
    x = np.asarray(image).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis].copy()


@app.get("/health")
async def health() -> dict[str, str]:
    client = app.state.client
    try:
        ready = await client.is_server_ready() and await client.is_model_ready(MODEL_NAME)
    except Exception:  # noqa: BLE001
        ready = False
    if not ready:
        raise HTTPException(status_code=503, detail="triton or model is not ready")
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile) -> dict[str, Any]:
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        x = preprocess(content)
    except OSError:
        raise HTTPException(status_code=400, detail="not an image") from None

    inputs = [triton.InferInput("input", list(x.shape), "FP32")]
    inputs[0].set_data_from_numpy(x)
    result = await app.state.client.infer(MODEL_NAME, inputs)
    logits = result.as_numpy("logits")[0]

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    class_id = int(probs.argmax())
    return {
        "class_id": class_id,
        "class_name": CLASSES[class_id],
        "probabilities": [float(p) for p in probs],
        "model_name": MODEL_NAME,
        "model_version": result.get_response()["model_version"],
    }
