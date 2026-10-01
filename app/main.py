"""FastAPI application: POST /mosaics and GET /mosaics/{id}."""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from . import __version__
from .db import Database
from .mosaic import MosaicError, build_mosaic
from .preview import render_preview
from .schemas import MosaicRequest, MosaicResult, TargetGrid

DB_PATH = os.environ.get("MOSAIC_DB_PATH", "/data/mosaics.db")

app = FastAPI(
    title="Multi-temporal Cloud-gap Mosaic API",
    version=__version__,
    description=(
        "Submit 2–16 co-registered raster slices with cloud masks and receive a "
        "deterministic cloud-gap mosaic: per-pixel provenance, coverage, hole "
        "diagnostics and overlap-conflict evidence."
    ),
)
db = Database(DB_PATH)


def _persist_and_respond(
    req: MosaicRequest, request_dict: dict[str, Any]
) -> dict[str, Any]:
    from datetime import datetime, timezone

    built = build_mosaic(req)
    input_summary = built["input_summary"]
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    result: dict[str, Any] = {
        "status": "completed",
        "request_id": req.request_id,
        "target": request_dict["target"],
        "reference_time": built["reference_time"]
        .isoformat()
        .replace("+00:00", "Z"),
        "max_time_delta_seconds": req.max_time_delta_seconds,
        "mosaic": built["mosaic"],
        "sources": built["sources"],
        "selected_index": built["selected_index"],
        "hole_mask": built["hole_mask"],
        "hole_reasons": built["hole_reasons"],
        "coverage_fraction": built["coverage_fraction"],
        "band_coverage_fraction": built["band_coverage_fraction"],
        "conflict_count": built["conflict_count"],
        "conflict_fraction": built["conflict_fraction"],
        "conflict_evidence": built["conflict_evidence"],
        "holes": built["holes"],
        "warnings": built["warnings"],
        "input_summary": input_summary,
        "created_at": created_at,
    }

    try:
        mosaic_id, reused, stored = db.create(
            request_id=req.request_id,
            request_dict=request_dict,
            input_summary=input_summary,
            result=result,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    result["id"] = mosaic_id
    if reused:
        stored["reused"] = True
        return stored
    result["reused"] = False
    return result


@app.exception_handler(MosaicError)
async def mosaic_error_handler(_request, exc: MosaicError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/mosaics", response_model=MosaicResult, tags=["mosaics"])
def create_mosaic(req: MosaicRequest) -> dict[str, Any]:
    request_dict = req.model_dump(mode="json")
    return _persist_and_respond(req, request_dict)


@app.get("/mosaics/{mosaic_id}", response_model=MosaicResult, tags=["mosaics"])
def get_mosaic(mosaic_id: str) -> dict[str, Any]:
    result = db.get(mosaic_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"mosaic {mosaic_id!r} not found")
    return result


@app.get("/mosaics/{mosaic_id}/preview.png", tags=["mosaics"])
def get_preview(mosaic_id: str) -> Response:
    result = db.get(mosaic_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"mosaic {mosaic_id!r} not found")
    png = render_preview(result)
    return Response(content=png, media_type="image/png")
