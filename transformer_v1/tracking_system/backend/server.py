"""FastAPI service for Transformer-based cooperative tracking playback."""

from __future__ import annotations

import asyncio
import csv
import contextlib
import io
import json
import logging
import platform
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from tracking_system.backend.batch_testing import BatchTestManager
from tracking_system.backend.evaluation import ACCEPTANCE_CRITERIA, EvaluationService
from tracking_system.backend.model_manager import ModelManager
from tracking_system.backend.time_utils import nearest_time_index


SERVICE_STARTED = time.monotonic()
LOGGER = logging.getLogger("tracking_system.backend")
MANAGER_INIT_LOCK = threading.RLock()
_MODEL_MANAGER: ModelManager | None = None
_BATCH_MANAGER: BatchTestManager | None = None
_EVALUATION_SERVICE: EvaluationService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if get_batch_test_manager.cache_info().currsize:
        get_batch_test_manager().shutdown()


app = FastAPI(
    title="End-to-End Cooperative Tracking API",
    version="1.4.0",
    description="Model deployment, batch testing and assessment service for CoopTrackerModel",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("Unhandled request error (request_id=%s)", request_id)
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
    LOGGER.info(
        "HTTP %s %s -> %s %.3f ms (request_id=%s)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


@lru_cache(maxsize=1)
def get_model_manager() -> ModelManager:
    global _MODEL_MANAGER
    with MANAGER_INIT_LOCK:
        if _MODEL_MANAGER is None:
            _MODEL_MANAGER = ModelManager()
        return _MODEL_MANAGER


@lru_cache(maxsize=1)
def get_batch_test_manager() -> BatchTestManager:
    global _BATCH_MANAGER
    with MANAGER_INIT_LOCK:
        if _BATCH_MANAGER is None:
            _BATCH_MANAGER = BatchTestManager(get_model_manager())
        return _BATCH_MANAGER


@lru_cache(maxsize=1)
def get_evaluation_service() -> EvaluationService:
    global _EVALUATION_SERVICE
    with MANAGER_INIT_LOCK:
        if _EVALUATION_SERVICE is None:
            _EVALUATION_SERVICE = EvaluationService(get_batch_test_manager())
        return _EVALUATION_SERVICE


class ActivateModelRequest(BaseModel):
    model_id: str = Field(min_length=1)
    device: str = Field(default="auto", min_length=1)


class CreateBatchTestRequest(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    model_id: str | None = None
    dataset_id: str = Field(default="v7", min_length=1)
    device: str = Field(default="auto", min_length=1)
    warmup_frames: int = Field(default=5, ge=0, le=99)
    batch_size: int = Field(default=64, ge=1, le=256)
    workers: int = Field(default=0, ge=0, le=8)
    max_scenarios: int | None = Field(default=None, ge=1)


@app.get("/api/health")
def health() -> dict:
    batch = (
        get_batch_test_manager().status()
        if get_batch_test_manager.cache_info().currsize
        else {"worker_count": 1, "jobs": {}, "active_job_ids": []}
    )
    return {
        "status": "ready",
        "service_version": app.version,
        "uptime_seconds": round(time.monotonic() - SERVICE_STARTED, 3),
        "active_model": get_model_manager().active_info(),
        "batch_testing": batch,
    }


@app.get("/api/system")
def system_status() -> dict:
    cuda_devices = get_model_manager().available_devices()
    return {
        "service": {
            "name": app.title,
            "version": app.version,
            "uptime_seconds": round(time.monotonic() - SERVICE_STARTED, 3),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
        },
        "devices": cuda_devices,
        "active_model": get_model_manager().active_info(),
        "batch_testing": get_batch_test_manager().status(),
    }


@app.get("/api/config/acceptance-criteria")
def acceptance_criteria() -> dict:
    result = EvaluationService.criteria()
    return {"total": len(result["items"]), **result}


@app.get("/api/devices")
def devices() -> dict:
    items = get_model_manager().available_devices()
    return {"items": items}


@app.get("/api/models")
def models() -> dict:
    items = get_model_manager().list_models()
    return {"total": len(items), "items": items}


@app.get("/api/models/active")
def active_model() -> dict:
    return get_model_manager().active_info()


@app.post("/api/models/refresh")
def refresh_models() -> dict:
    items = get_model_manager().refresh()
    return {"total": len(items), "items": items}


@app.post("/api/models/activate")
async def activate_model(request: ActivateModelRequest) -> dict:
    manager = get_model_manager()
    try:
        return await asyncio.to_thread(
            manager.activate, request.model_id, request.device
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/scenarios")
def scenarios(limit: int | None = Query(default=None, ge=1)) -> dict:
    items = get_model_manager().get_engine().list_scenarios()
    return {"total": len(items), "items": items[:limit] if limit else items}


@app.get("/api/datasets")
def datasets() -> dict:
    items = get_batch_test_manager().list_datasets()
    return {"total": len(items), "items": items}


@app.post("/api/batch-tests", status_code=202)
def create_batch_test(request: CreateBatchTestRequest) -> dict:
    try:
        return get_batch_test_manager().create_job(**request.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/batch-tests")
def batch_tests(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    items = get_batch_test_manager().list_jobs(limit=limit)
    return {"total": len(items), "items": items}


@app.get("/api/batch-tests/{job_id}")
def batch_test(job_id: str) -> dict:
    try:
        return get_batch_test_manager().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/batch-tests/{job_id}/cancel")
def cancel_batch_test(job_id: str) -> dict:
    try:
        return get_batch_test_manager().cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/batch-tests/{job_id}/assessment")
def batch_test_assessment(
    job_id: str,
    baseline_rmse_m: float | None = Query(default=None, gt=0),
) -> dict:
    try:
        return get_evaluation_service().assess_job(job_id, baseline_rmse_m)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/batch-tests/{job_id}/evidence/{metric_id}")
async def generate_batch_test_evidence(job_id: str, metric_id: str) -> dict:
    try:
        return await asyncio.to_thread(
            get_evaluation_service().generate_evidence, job_id, metric_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/batch-tests/{job_id}/report.json")
def download_batch_report(job_id: str) -> FileResponse:
    try:
        job = get_batch_test_manager().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = get_batch_test_manager().report_dir / f"{job_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Batch report file is unavailable")
    safe_name = f"batch-test-{job.get('config', {}).get('model_id', 'model')}-{job_id[:8]}.json"
    return FileResponse(path, media_type="application/json", filename=safe_name)


@app.get("/api/batch-tests/{job_id}/tracks.csv")
def download_track_metrics(job_id: str) -> StreamingResponse:
    try:
        job = get_batch_test_manager().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tracks = (job.get("result") or {}).get("tracks") or []
    if job.get("status") != "completed" or not tracks:
        raise HTTPException(status_code=409, detail="Completed per-track results are unavailable")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["track", "valid_frames", "rmse_m", "mae_m", "p95_m", "max_error_m"],
    )
    writer.writeheader()
    writer.writerows(tracks)
    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment; filename="batch-test-{job_id[:8]}-tracks.csv"'
    return response


@app.get("/api/evaluations/references")
def evaluation_references() -> dict:
    return get_evaluation_service().references()


@app.get("/api/evaluations/cross-dataset")
def cross_dataset_evaluation() -> dict:
    return get_evaluation_service().cross_dataset()


@app.get("/api/evaluations/compare")
def compare_batch_tests(
    job_ids: str | None = Query(
        default=None,
        description="Optional comma-separated completed batch test ids",
    ),
) -> dict:
    selected = None
    if job_ids:
        selected = [token.strip() for token in job_ids.split(",") if token.strip()]
        if not selected:
            raise HTTPException(status_code=422, detail="job_ids must contain at least one id")
        if len(selected) > 50:
            raise HTTPException(status_code=422, detail="At most 50 jobs can be compared")
    try:
        return get_evaluation_service().compare_jobs(selected)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/reports")
def report_catalog(limit: int = Query(default=200, ge=1, le=200)) -> dict:
    return get_evaluation_service().report_catalog(limit)


@app.get("/api/reports/comparison.csv")
def download_comparison_report(
    job_ids: str = Query(min_length=1, description="Comma-separated completed job ids"),
) -> StreamingResponse:
    selected = [token.strip() for token in job_ids.split(",") if token.strip()]
    if not selected:
        raise HTTPException(status_code=422, detail="job_ids must contain at least one id")
    if len(selected) > 50:
        raise HTTPException(status_code=422, detail="At most 50 jobs can be exported")
    try:
        rows = get_evaluation_service().compare_jobs(selected)["items"]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not rows:
        raise HTTPException(status_code=409, detail="No completed jobs were selected")
    output = io.StringIO(newline="")
    fieldnames = [
        "job_id", "name", "model_id", "dataset_id", "evaluated_scenarios",
        "evaluated_tracks", "position_rmse_m", "position_mae_m", "position_p95_m",
        "estimated_frame_ms", "throughput_tracks_per_second", "association_accuracy",
        "created_at", "finished_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)
    response = StreamingResponse(
        iter(["\ufeff" + output.getvalue()]), media_type="text/csv; charset=utf-8"
    )
    response.headers["Content-Disposition"] = 'attachment; filename="tracking-comparison.csv"'
    return response


@app.get("/api/reports/{job_id}/assessment.json")
def download_assessment_report(
    job_id: str,
    baseline_rmse_m: float | None = Query(default=None, gt=0),
) -> StreamingResponse:
    try:
        bundle = get_evaluation_service().assessment_bundle(job_id, baseline_rmse_m)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    payload = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    model_id = bundle["assessment"].get("model_id") or "model"
    response = StreamingResponse(iter([payload]), media_type="application/json; charset=utf-8")
    response.headers["Content-Disposition"] = (
        f'attachment; filename="assessment-{model_id}-{job_id[:8]}.json"'
    )
    return response


@app.websocket("/ws/tracking")
async def tracking_endpoint(
    websocket: WebSocket,
    dataset: str = "100",
    target: int = Query(default=0, ge=0),
) -> None:
    await websocket.accept()
    receive_task: asyncio.Task | None = None
    try:
        engine = get_model_manager().get_engine()
        sequence = await asyncio.to_thread(engine.infer_scenario, dataset, target)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        await websocket.close(code=1008, reason=str(exc)[:120])
        return

    control = {"paused": False, "index": 0, "seek_target": None}

    async def receive_commands() -> None:
        try:
            while True:
                message = await websocket.receive_json()
                action = message.get("action")
                if action == "pause":
                    control["paused"] = True
                elif action == "play":
                    control["paused"] = False
                elif action == "seek":
                    control["seek_target"] = float(message.get("time", 0.0))
        except (WebSocketDisconnect, RuntimeError, ValueError, TypeError):
            return

    receive_task = asyncio.create_task(receive_commands())
    errors_since_reset: list[float] = []

    try:
        while control["index"] < sequence.frame_count:
            seek_target = control["seek_target"]
            if seek_target is not None:
                control["index"] = nearest_time_index(sequence.times, seek_target)
                control["seek_target"] = None
                errors_since_reset.clear()

            if control["paused"]:
                await asyncio.sleep(0.1)
                continue

            index = control["index"]
            error = float(sequence.position_errors[index])
            errors_since_reset.append(error)
            running_rmse = float(np.sqrt(np.mean(np.square(errors_since_reset))))
            truth_position = sequence.truth_states[index, [0, 2, 4]]
            predicted_position = sequence.predicted_states[index, [0, 2, 4]]

            await websocket.send_json(
                {
                    "time": float(sequence.times[index]),
                    "max_time": sequence.max_time,
                    "is_reset": len(errors_since_reset) == 1,
                    "scenario": sequence.scenario_name,
                    "target_index": sequence.target_index,
                    "truth_pos": truth_position.tolist(),
                    "pred_pos": predicted_position.tolist(),
                    "metrics": {
                        "err_pos": error,
                        "rmse_current": running_rmse,
                        "inference_time_ms": sequence.per_frame_inference_ms,
                        "sequence_inference_time_ms": sequence.sequence_inference_ms,
                    },
                }
            )
            control["index"] += 1
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if receive_task is not None:
            receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
