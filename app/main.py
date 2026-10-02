"""多时相遥感云隙拼接 API。

启动：``uvicorn app.main:app --host 0.0.0.0 --port 8000``
交互式文档：``/docs``（Swagger UI）、``/redoc``
"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from . import db
from .models import MosaicOut, MosaicRequest
from .mosaic_service import MosaicError, build_mosaic


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="多时相遥感云隙拼接 API",
    description=(
        "提交同一坐标网格的 2～16 个时相栅格切片（含采集时间、分辨率、左上原点、"
        "波段矩阵、无数据值与云掩膜），服务按仿射位置落入输出网格，逐像元排除云与"
        "无数据，按 **云量 → 时间接近度 → 边缘距离 → 请求序号** 的固定优先级选源，"
        "检测重叠冲突与无法覆盖的空洞，返回确定性拼接结果。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(MosaicError)
async def mosaic_error_handler(request: Request, exc: MosaicError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/healthz", tags=["meta"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/mosaics",
    response_model=MosaicOut,
    status_code=201,
    tags=["mosaics"],
    summary="创建云隙拼接任务（相同请求体重试为幂等返回）",
    responses={409: {"description": "request_id 已用于不同请求体"}},
)
def create_mosaic(request: MosaicRequest, response: Response) -> MosaicOut:
    payload = request.model_dump()
    request_hash = db.canonical_hash(payload)

    existing = db.find_by_hash(request_hash)
    if existing is not None:
        response.status_code = 200
        return MosaicOut(**existing)

    if request.request_id is not None:
        other = db.find_by_request_id(request.request_id)
        if other is not None and other.get("request_hash") != request_hash:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"request_id={request.request_id!r} 已对应另一拼接结果 "
                    f"({other.get('id')})；相同 request_id 的请求体必须一致"
                ),
            )

    result = build_mosaic(request, request_hash)

    try:
        db.save_mosaic(
            mosaic_id=result.id,
            request_hash=request_hash,
            request_id=request.request_id,
            payload=payload,
            result=result.model_dump(),
        )
    except sqlite3.IntegrityError:
        # 并发幂等：另一个请求已落库，重读已有结果
        existing = db.find_by_hash(request_hash)
        if existing is None and request.request_id is not None:
            other = db.find_by_request_id(request.request_id)
            if other is not None:
                response.status_code = 200
                return MosaicOut(**other)
            raise HTTPException(status_code=409, detail="request_id 冲突")
        if existing is not None:
            response.status_code = 200
            return MosaicOut(**existing)
        raise
    except sqlite3.OperationalError as exc:  # 锁竞争兜底
        existing = db.find_by_hash(request_hash)
        if existing is not None:
            response.status_code = 200
            return MosaicOut(**existing)
        raise HTTPException(status_code=503, detail=f"数据库暂时不可用: {exc}") from exc

    return result


@app.get(
    "/mosaics/{mosaic_id}",
    response_model=MosaicOut,
    tags=["mosaics"],
    summary="按 ID 获取拼接结果（持久化，支持幂等复核）",
    responses={404: {"description": "任务不存在"}},
)
def get_mosaic(mosaic_id: str) -> MosaicOut:
    result = db.get_mosaic(mosaic_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"拼接任务 {mosaic_id} 不存在")
    return MosaicOut(**result)
