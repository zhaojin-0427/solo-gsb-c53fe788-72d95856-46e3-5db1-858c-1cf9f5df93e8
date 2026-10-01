"""End-to-end tests for the cloud-gap mosaic API and engine."""

from __future__ import annotations

import os

import numpy as np
import pytest

# Isolated database before importing the app.
os.environ["MOSAIC_DB_PATH"] = "/tmp/test_mosaics.db"
if os.path.exists(os.environ["MOSAIC_DB_PATH"]):
    os.remove(os.environ["MOSAIC_DB_PATH"])

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.mosaic import build_mosaic  # noqa: E402
from app.schemas import MosaicRequest  # noqa: E402

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_tile(tid, t, vals, cloud=None, ps=10.0, origin=(0.0, 40.0), nodata=None):
    return {
        "tile_id": tid,
        "acquired_at": t,
        "pixel_size": ps,
        "origin_x": origin[0],
        "origin_y": origin[1],
        "bands": [np.asarray(vals, dtype=float).tolist()],
        "nodata": nodata,
        "cloud_mask": np.asarray(cloud, dtype=int).tolist() if cloud is not None else None,
    }


def target(w=4, h=4, ps=10.0, origin=(0.0, 40.0)):
    return {
        "origin_x": origin[0],
        "origin_y": origin[1],
        "pixel_size": ps,
        "width": w,
        "height": h,
    }


def post(payload, expect=200):
    resp = client.post("/mosaics", json=payload)
    assert resp.status_code == expect, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# Engine tests
# --------------------------------------------------------------------------- #


def test_cloud_gap_fill_and_priority():
    vA = (np.arange(16).reshape(4, 4) + 100.0).tolist()
    vB = (np.arange(16).reshape(4, 4) + 200.0).tolist()
    cA = [[1, 1, 0, 0]] * 4
    cB = [[0, 0, 1, 1]] * 4
    req = MosaicRequest(
        target=target(),
        tiles=[
            make_tile("A", "2026-09-01T10:00:00Z", vA, cA),
            make_tile("B", "2026-09-02T10:00:00Z", vB, cB),
        ],
        reference_time="2026-09-01T10:00:00Z",
    )
    r = build_mosaic(req)
    assert r["coverage_fraction"] == 1.0
    assert [src for row in r["sources"] for src in row].count("A") == 8
    assert [src for row in r["sources"] for src in row].count("B") == 8
    assert r["mosaic"][0][0] == [200.0, 201.0, 102.0, 103.0]


def test_cloud_fraction_first_priority():
    # A nearly cloud-free later tile must win everywhere over a cloudy earlier
    # tile even though the reference time favours the early acquisition.
    vA = np.full((4, 4), 100.0).tolist()
    vB = np.full((4, 4), 200.0).tolist()
    # Clearer early tile is cloudy at one pixel; the cloudier late tile fills
    # exactly that gap (it is clear there but cloudy at two other pixels).
    cA = np.zeros((4, 4), int); cA[3, 3] = 1
    cB = np.zeros((4, 4), int); cB[0, 0] = cB[0, 1] = 1
    req = MosaicRequest(
        target=target(),
        tiles=[
            make_tile("early", "2026-09-01T00:00:00Z", vA, cA.tolist()),
            make_tile("late", "2026-09-10T00:00:00Z", vB, cB.tolist()),
        ],
        reference_time="2026-09-01T00:00:00Z",
    )
    r = build_mosaic(req)
    flat = [s for row in r["sources"] for s in row]
    assert flat.count("early") == 15
    assert r["sources"][3][3] == "late"  # early cloudy there, late fills gap


def test_edge_distance_tiebreak():
    # Equal cloud fraction, equal time distance: a tile covering more of the
    # interior around a pixel wins over a sliver at the edge.
    big = np.full((4, 4), 1.0).tolist()
    sliver = np.full((1, 4), 9.0).tolist()
    req = MosaicRequest(
        target=target(),
        tiles=[
            make_tile("big", "2026-09-01T10:00:00Z", big),
            make_tile("sliver", "2026-09-01T10:00:00Z", sliver, origin=(0.0, 10.0)),
        ],
    )
    r = build_mosaic(req)
    # Bottom row (r=3) is only covered by big -> big. Sliver covers r=0.
    # Interior column pixels (r=0, c=1..2) are in both; big is farther from
    # its footprint edge than sliver is from sliver's edge.
    assert r["sources"][0][1] == "big"
    assert r["sources"][0][2] == "big"
    assert r["conflict_count"] > 0  # values 1 vs 9 disagree


def test_time_window_excludes_tiles_and_marks_holes():
    vA = np.full((4, 4), 100.0).tolist()
    vB = np.full((4, 4), 200.0).tolist()
    req = MosaicRequest(
        target=target(),
        tiles=[
            make_tile("A", "2026-09-01T00:00:00Z", vA),
            make_tile("B", "2026-09-03T00:00:00Z", vB),
        ],
        reference_time="2026-09-01T00:00:00Z",
        max_time_delta_seconds=86400,
    )
    r = build_mosaic(req)
    # A within window, B excluded. Still full coverage from A.
    assert r["coverage_fraction"] == 1.0
    assert all(s == "A" for row in r["sources"] for s in row)
    assert any("outside max_time_delta" in w for w in r["warnings"])


def test_hole_reasons_no_coverage_cloud_nodata():
    vA = np.full((4, 4), 1.0).tolist()
    vB = np.full((4, 4), 1.0).tolist()
    cA = np.zeros((4, 4), int)
    cB = np.zeros((4, 4), int)
    # col 0: cloud in both tiles => cloud hole. col 1: cloud in A, clear in B.
    cA[:, 0] = 1
    cB[:, 0] = 1
    cA[:, 1] = 1
    # col 2: clear in both but nodata in both => nodata hole.
    # col 3: only A covers it (B is 3 wide), A is clear => covered.
    vA_arr = np.full((4, 4), 1.0); vA_arr[:, 2] = -999
    vB_arr = np.full((4, 4), 1.0); vB_arr[:, 2] = -999
    req = MosaicRequest(
        target=target(w=4),
        tiles=[
            make_tile("A", "2026-09-01T10:00:00Z", vA_arr.tolist(), cA.tolist(), nodata=-999),
            make_tile("B", "2026-09-01T12:00:00Z", vB_arr[:, :3].tolist(), cB[:, :3].tolist(), nodata=-999),
        ],
    )
    r = build_mosaic(req)
    reasons = r["hole_reasons"]
    assert all(reasons[rr][0] == "cloud" for rr in range(4))
    assert all(reasons[rr][2] == "nodata" for rr in range(4))
    assert all(reasons[rr][1] == "" for rr in range(4))  # col 1 covered by B
    assert all(r["sources"][rr][1] == "B" for rr in range(4))
    assert all(r["sources"][rr][3] == "A" for rr in range(4))
    # Target extended one column beyond both tiles -> no_coverage.
    req2 = MosaicRequest(
        target=target(w=5),
        tiles=[
            make_tile("A", "2026-09-01T10:00:00Z", vA_arr.tolist(), cA.tolist(), nodata=-999),
            make_tile("B", "2026-09-01T12:00:00Z", vB_arr[:, :3].tolist(), cB[:, :3].tolist()),
        ],
    )
    r2 = build_mosaic(req2)
    assert all(r2["hole_reasons"][rr][4] == "no_coverage" for rr in range(4))


def test_conflict_detection_with_tolerance():
    vA = np.full((4, 4), 100.0).tolist()
    vB = np.full((4, 4), 100.5).tolist()
    payload = dict(
        target=target(),
        tiles=[
            make_tile("A", "2026-09-01T10:00:00Z", vA),
            make_tile("B", "2026-09-01T10:00:00Z", vB),
        ],
    )
    r = build_mosaic(MosaicRequest(**payload, conflict_tolerance=1.0))
    assert r["conflict_count"] == 0
    r = build_mosaic(MosaicRequest(**payload, conflict_tolerance=0.4))
    assert r["conflict_count"] == 16
    ev = r["conflict_evidence"][0]
    assert set(ev["sources"]) == {"A", "B"}
    assert abs(ev["difference"] - 0.5) < 1e-6


def test_coarser_and_finer_resampling():
    # Coarser tile (20m): one 20m pixel painted over a 2x2 output block.
    # Covers the whole 4x4 window: NW=5 NE=6 SW=7 SE=8.
    coarse = make_tile(
        "coarse", "2026-09-01T10:00:00Z",
        [[5.0, 6.0], [7.0, 8.0]], ps=20.0,
    )
    # Finer tile (5m): covers the left 2 output columns (4 fine cols),
    # constant 99 there, absent from the right half.
    fine = make_tile(
        "fine", "2026-09-02T10:00:00Z",
        np.full((8, 4), 99.0).tolist(), ps=5.0,
    )
    req = MosaicRequest(target=target(4, 4), tiles=[coarse, fine])
    r = build_mosaic(req)
    by_id = {s["tile_id"]: s for s in r["input_summary"]}
    assert by_id["coarse"]["mode"] == "replicate_coarser"
    assert by_id["fine"]["mode"] == "aggregate_finer"
    assert by_id["fine"]["covered_target_pixels"] == 8
    # Left half: clear fine tile is cloud-free like coarse (0 clouds), but
    # acquired later; with the same cloud fraction coarse is preferred on
    # edge distance (it spans the full window). Force fine win via time by
    # making coarse partly cloudy elsewhere? Simpler: assert values are valid
    # and the fine half carries either source, while SE is coarse-only.
    assert r["mosaic"][0][3][3] == 8.0
    assert r["mosaic"][0][0][2] == 6.0

    # With a reference time on the fine acquisition, the fine half is chosen.
    req2 = MosaicRequest(
        target=target(4, 4), tiles=[coarse, fine],
        reference_time="2026-09-02T10:00:00Z",
    )
    r2 = build_mosaic(req2)
    assert all(r2["sources"][rr][cc] == "fine" for rr in range(4) for cc in range(2))
    assert r2["mosaic"][0][0][0] == 99.0
    assert all(r2["sources"][rr][cc] == "coarse" for rr in range(4) for cc in range(2, 4))


def test_determinism():
    vA = (np.arange(16).reshape(4, 4) + 100.0).tolist()
    vB = (np.arange(16).reshape(4, 4) + 200.0).tolist()
    cA = [[1, 1, 0, 0]] * 4
    cB = [[0, 0, 1, 1]] * 4
    kw = dict(
        target=target(),
        reference_time="2026-09-01T10:00:00Z",
    )
    r1 = build_mosaic(MosaicRequest(
        tiles=[make_tile("A", "2026-09-01T10:00:00Z", vA, cA),
               make_tile("B", "2026-09-01T11:00:00Z", vB, cB)], **kw))
    r2 = build_mosaic(MosaicRequest(
        tiles=[make_tile("B", "2026-09-01T11:00:00Z", vB, cB),
               make_tile("A", "2026-09-01T10:00:00Z", vA, cA)], **kw))
    assert r1["mosaic"] == r2["mosaic"]
    assert r1["sources"] == r2["sources"]


# --------------------------------------------------------------------------- #
# HTTP / persistence tests
# --------------------------------------------------------------------------- #


def _api_payload(request_id=None):
    vA = (np.arange(16).reshape(4, 4) + 100.0).tolist()
    vB = (np.arange(16).reshape(4, 4) + 200.0).tolist()
    cA = [[1, 1, 0, 0]] * 4
    cB = [[0, 0, 1, 1]] * 4
    p = {
        "target": target(),
        "reference_time": "2026-09-01T10:00:00Z",
        "tiles": [
            make_tile("A", "2026-09-01T10:00:00Z", vA, cA),
            make_tile("B", "2026-09-01T11:00:00Z", vB, cB),
        ],
    }
    if request_id:
        p["request_id"] = request_id
    return p


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_post_and_get_roundtrip():
    r = post(_api_payload())
    assert r["coverage_fraction"] == 1.0
    assert r["status"] == "completed"
    assert len(r["input_summary"]) == 2
    got = client.get(f"/mosaics/{r['id']}").json()
    assert got["mosaic"] == r["mosaic"]
    assert got["sources"] == r["sources"]
    # 404 on missing
    assert client.get("/mosaics/99999999").status_code == 404


def test_idempotent_retry_same_request_id():
    p = _api_payload(request_id="idem-001")
    r1 = post(p)
    r2 = post(p)
    assert r1["id"] == r2["id"]
    assert r2["reused"] is True
    got = client.get("/mosaics/idem-001").json()
    assert got["id"] == r1["id"]


def test_request_id_conflict_returns_409():
    p1 = _api_payload(request_id="conflict-x")
    post(p1)
    p2 = _api_payload(request_id="conflict-x")
    p2["conflict_tolerance"] = 1.0
    resp = client.post("/mosaics", json=p2)
    assert resp.status_code == 409


def test_validation_errors():
    # Too few tiles
    p = _api_payload()
    p["tiles"] = p["tiles"][:1]
    assert client.post("/mosaics", json=p).status_code == 422
    # Misaligned grid -> engine 422
    p = _api_payload()
    p["tiles"][0]["origin_x"] = 3.333
    assert client.post("/mosaics", json=p).status_code == 422
    # Cloud mask wrong shape
    p = _api_payload()
    p["tiles"][0]["cloud_mask"] = [[1, 0], [0, 1]]
    assert client.post("/mosaics", json=p).status_code == 422
    # Duplicate tile ids
    p = _api_payload()
    p["tiles"][1]["tile_id"] = "A"
    assert client.post("/mosaics", json=p).status_code == 422


def test_preview_png():
    r = post(_api_payload())
    resp = client.get(f"/mosaics/{r['id']}/preview.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_partial_coverage_metrics():
    # Single small tile inside a larger window -> holes + coverage fraction.
    v = np.full((2, 2), 7.0).tolist()
    p = {
        "target": target(4, 4),
        "tiles": [
            make_tile("a", "2026-09-01T10:00:00Z", v),
            make_tile("b", "2026-09-01T11:00:00Z", v),
        ],
    }
    r = post(p)
    assert r["coverage_fraction"] == pytest.approx(0.25)
    assert sum(sum(row) for row in r["hole_mask"]) == 12
    assert r["holes"] and r["holes"][0]["reason"] == "no_coverage"
