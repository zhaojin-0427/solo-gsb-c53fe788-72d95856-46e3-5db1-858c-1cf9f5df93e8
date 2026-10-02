"""端到端接口测试。"""
from __future__ import annotations

import copy

import numpy as np


PIXEL = (10.0, -10.0)
ORIGIN = {"x": 0.0, "y": 100.0}


def make_tile(
    tile_id: str,
    acquired_at: str,
    array: np.ndarray,
    cloud: np.ndarray | None = None,
    origin=None,
    nodata=-9999.0,
):
    h, w = array.shape[1:]
    if cloud is None:
        cloud = np.zeros((h, w), dtype=np.uint8)
    return {
        "id": tile_id,
        "acquired_at": acquired_at,
        "pixel_size": list(PIXEL),
        "origin": origin or copy.deepcopy(ORIGIN),
        "nodata": nodata,
        "bands": array.tolist(),
        "cloud_mask": cloud.tolist(),
    }


def base_target(w=4, h=4):
    return {
        "origin": copy.deepcopy(ORIGIN),
        "width": w,
        "height": h,
        "pixel_size": list(PIXEL),
        "nodata": -9999.0,
    }


def post(client, payload, expected=201):
    resp = client.post("/mosaics", json=payload)
    assert resp.status_code == expected, resp.text
    return resp.json()


# ---------------------------------------------------------------------


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_complementary_cloud_gap_fill(client):
    """两个时相云区互补，应 100% 覆盖且各像元选择云少/时间近的来源。"""
    a = np.full((1, 4, 4), 100, dtype=np.int32)
    b = np.full((1, 4, 4), 200, dtype=np.int32)
    cloud_a = np.zeros((4, 4), np.uint8)
    cloud_a[:, 2:] = 1  # 右半云
    cloud_b = np.zeros((4, 4), np.uint8)
    cloud_b[:, :2] = 1  # 左半云
    payload = {
        "tiles": [
            make_tile("t1", "2026-09-01T10:00:00Z", a, cloud_a),
            make_tile("t2", "2026-09-02T10:00:00Z", b, cloud_b),
        ],
        "target": base_target(),
        "max_time_delta_seconds": 86400 * 30,
    }
    out = post(client, payload)
    assert out["coverage"]["covered_fraction"] == 1.0
    assert out["coverage"]["hole_pixels"] == 0
    bands = np.array(out["bands"])
    # 左半仅 t1 清晰（t2 有云）；右半仅 t2 清晰（t1 有云）
    assert np.array_equal(bands[0, :, :2], np.full((4, 2), 100))
    assert np.array_equal(bands[0, :, 2:], np.full((4, 2), 200))
    src = np.array(out["source_map"])
    assert np.array_equal(src[:, :2], np.zeros((4, 2)))
    assert np.array_equal(src[:, 2:], np.ones((4, 2)))
    assert out["source_ids"] == ["t1", "t2"]
    assert out["reference_time"] == "2026-09-02T10:00:00Z"
    assert out["tiles"][0]["used_pixels"] == 8
    assert out["tiles"][1]["used_pixels"] == 8
    # 两时相几何完全重叠（各自 8 个清晰像元都落在对方范围内）
    assert out["overlap"]["pixels_with_multiple_candidates"] == 0


def test_nodata_and_uncoverable_hole(client):
    """无数据像元被排除；两者都缺的区域形成空洞并给出连通分量。"""
    a = np.full((1, 4, 4), 10, dtype=np.int32)
    a[0, 0, 0] = -9999
    b = np.full((1, 4, 4), 20, dtype=np.int32)
    b[0, 0, 0] = -9999
    b[0, 3, 3] = -9999
    payload = {
        "tiles": [
            make_tile("t1", "2026-09-01T10:00:00Z", a),
            make_tile("t2", "2026-09-01T11:00:00Z", b),
        ],
        "target": base_target(),
        "max_time_delta_seconds": 3600,
    }
    out = post(client, payload)
    assert out["coverage"]["covered_pixels"] == 14
    assert out["coverage"]["hole_pixels"] == 2
    assert out["bands"][0][0][0] == -9999.0
    assert out["source_map"][0][0] == -1
    # 一个 2 像元的 4 连通空洞（(0,0) 与 (3,3) 不连通 -> 两个分量）
    assert len(out["coverage"]["holes"]) == 2
    assert out["coverage"]["holes"][0]["size"] == 1
    assert out["bands"][0][3][3] == -9999.0


def test_max_time_delta_excludes_tile(client):
    """超出最大时间差的时相不参与选源，空洞由剩余时相决定。"""
    a = np.full((1, 2, 2), 1, dtype=np.int32)
    b = np.full((1, 2, 2), 2, dtype=np.int32)
    cloud_b = np.zeros((2, 2), np.uint8)
    cloud_b[0, 0] = 1
    payload = {
        "tiles": [
            make_tile("old", "2026-01-01T00:00:00Z", a),
            make_tile("new", "2026-09-02T00:00:00Z", b, cloud_b),
        ],
        "target": base_target(2, 2),
        "max_time_delta_seconds": 3600,  # old 被排除；new 有云 -> 空洞
    }
    out = post(client, payload)
    assert out["tiles"][0]["excluded"] is True
    assert "max_time_delta" in out["tiles"][0]["exclude_reason"]
    assert out["tiles"][1]["excluded"] is False
    assert out["coverage"]["hole_pixels"] == 1
    assert out["source_map"][0][0] == -1
    assert out["bands"][0][0][1] == 2


def test_overlap_conflict_evidence(client):
    """重叠区两时相数值不一致，按容差记录冲突证据。"""
    a = np.full((1, 2, 2), 100, dtype=np.int32)
    b = np.full((1, 2, 2), 100, dtype=np.int32)
    b[0, 0, 0] = 130  # 差异 30
    payload = {
        "tiles": [
            make_tile("t1", "2026-09-01T10:00:00Z", a),
            make_tile("t2", "2026-09-02T10:00:00Z", b),
        ],
        "target": base_target(2, 2),
        "max_time_delta_seconds": 86400 * 10,
        "conflict_value_tolerance": 10,
    }
    out = post(client, payload)
    assert out["overlap"]["pixels_with_multiple_candidates"] == 4
    assert out["overlap"]["conflicting_pixels"] == 1
    ev = out["overlap"]["evidence"]
    assert len(ev) == 1
    assert ev[0]["row"] == 0 and ev[0]["col"] == 0
    assert ev[0]["winner"] == "t2"
    assert ev[0]["runner_up"] == "t1"
    assert ev[0]["winner_values"] == [130.0]
    assert ev[0]["abs_difference"] == [30.0]

    # 容差放大到 50 -> 不再算冲突
    payload["conflict_value_tolerance"] = 50
    out2 = post(client, payload)
    assert out2["overlap"]["conflicting_pixels"] == 0
    assert out2["overlap"]["evidence"] == []


def test_cloud_fraction_priority(client):
    """云量少的时相在非云区优先于更新但云更多的时相。"""
    a = np.full((1, 2, 2), 11, dtype=np.int32)  # 旧，但无云
    b = np.full((1, 2, 2), 22, dtype=np.int32)  # 新，但 75% 云
    cloud_b = np.ones((2, 2), np.uint8)
    cloud_b[0, 0] = 0  # 只剩一格非云
    payload = {
        "tiles": [
            make_tile("t1", "2026-09-01T00:00:00Z", a),
            make_tile("t2", "2026-09-02T00:00:00Z", b, cloud_b),
        ],
        "target": base_target(2, 2),
        "max_time_delta_seconds": 86400,
    }
    out = post(client, payload)
    bands = np.array(out["bands"])
    # t2 云量 0.75 > t1 云量 0，故清晰像元(0,0)也选 t1
    assert np.array_equal(bands[0], np.full((2, 2), 11))


def test_edge_distance_priority_and_shift(client):
    """云量、时间相同的情况下，优先使用切片内部像元（边缘距离）。

    两个 4 列宽切片在目标列 1 重叠：
    * t1 覆盖目标列 1..4，目标列 1 是其局部第 0 列（边缘，距离 0）；
    * t2 覆盖目标列 -2..1，目标列 1 是其局部第 3 列（内部，距离 3）。
    其余维度（云量 0、同时相）相同，故 (0,1) 必须选 t2。
    """
    a = np.full((1, 1, 4), 55, dtype=np.int32)
    b = np.full((1, 1, 4), 77, dtype=np.int32)
    origin_a = {"x": 10.0, "y": 100.0}   # 起始列 1
    origin_b = {"x": -20.0, "y": 100.0}  # 起始列 -2
    payload = {
        "tiles": [
            make_tile("t1", "2026-09-01T10:00:00Z", a, origin=origin_a),
            make_tile("t2", "2026-09-01T10:00:00Z", b, origin=origin_b),
        ],
        "target": base_target(3, 1),
        "max_time_delta_seconds": 3600,
    }
    out = post(client, payload)
    bands = np.array(out["bands"])
    src = np.array(out["source_map"])
    # col0 仅 t2；col1 两者重叠，t2 距边缘更远 -> 选 t2；col2 仅 t1
    assert bands[0, 0, 0] == 77
    assert bands[0, 0, 1] == 77
    assert src[0, 1] == 1
    assert bands[0, 0, 2] == 55
    assert src[0, 2] == 0
    assert out["coverage"]["covered_fraction"] == 1.0
    assert out["overlap"]["pixels_with_multiple_candidates"] == 1



def test_idempotent_retry_and_get(client):
    a = np.full((1, 2, 2), 1, dtype=np.int32)
    b = np.full((1, 2, 2), 2, dtype=np.int32)
    payload = {
        "tiles": [
            make_tile("t1", "2026-09-01T10:00:00Z", a),
            make_tile("t2", "2026-09-02T10:00:00Z", b),
        ],
        "target": base_target(2, 2),
        "max_time_delta_seconds": 86400,
        "request_id": "job-001",
    }
    r1 = client.post("/mosaics", json=payload)
    assert r1.status_code == 201
    r2 = client.post("/mosaics", json=payload)
    assert r2.status_code == 200
    j1, j2 = r1.json(), r2.json()
    assert j1["id"] == j2["id"]
    assert j1["request_hash"] == j2["request_hash"]

    got = client.get(f"/mosaics/{j1['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == j1["id"]

    missing = client.get("/mosaics/does-not-exist")
    assert missing.status_code == 404


def test_request_id_conflict_returns_409(client):
    a = np.full((1, 2, 2), 1, dtype=np.int32)
    b = np.full((1, 2, 2), 2, dtype=np.int32)
    common = {
        "target": base_target(2, 2),
        "max_time_delta_seconds": 86400,
        "request_id": "job-X",
    }
    p1 = {**common, "tiles": [
        make_tile("t1", "2026-09-01T10:00:00Z", a),
        make_tile("t2", "2026-09-02T10:00:00Z", b),
    ]}
    assert client.post("/mosaics", json=p1).status_code == 201
    p2 = {**common, "tiles": [
        make_tile("t1", "2026-09-01T10:00:00Z", a),
        make_tile("t3", "2026-09-03T10:00:00Z", b),
    ]}
    r = client.post("/mosaics", json=p2)
    assert r.status_code == 409


def test_validation_errors(client):
    good = make_tile("t1", "2026-09-01T10:00:00Z", np.full((1, 2, 2), 1, np.int32))
    # 少于 2 个切片
    bad = {"tiles": [good], "target": base_target(2, 2), "max_time_delta_seconds": 3600}
    assert client.post("/mosaics", json=bad).status_code == 422

    # 重复 id
    p = {
        "tiles": [
            make_tile("dup", "2026-09-01T10:00:00Z", np.full((1, 2, 2), 1, np.int32)),
            make_tile("dup", "2026-09-02T10:00:00Z", np.full((1, 2, 2), 2, np.int32)),
        ],
        "target": base_target(2, 2),
        "max_time_delta_seconds": 3600,
    }
    assert client.post("/mosaics", json=p).status_code == 422

    # 分辨率不一致
    t = make_tile("t2", "2026-09-02T10:00:00Z", np.full((1, 2, 2), 2, np.int32))
    t["pixel_size"] = [5.0, -5.0]
    p = {
        "tiles": [
            make_tile("t1", "2026-09-01T10:00:00Z", np.full((1, 2, 2), 1, np.int32)),
            t,
        ],
        "target": base_target(2, 2),
        "max_time_delta_seconds": 3600,
    }
    assert client.post("/mosaics", json=p).status_code == 422

    # 非整数像元偏移
    t2 = make_tile(
        "t2", "2026-09-02T10:00:00Z", np.full((1, 2, 2), 2, np.int32),
        origin={"x": 3.0, "y": 100.0},
    )
    p = {
        "tiles": [
            make_tile("t1", "2026-09-01T10:00:00Z", np.full((1, 2, 2), 1, np.int32)),
            t2,
        ],
        "target": base_target(2, 2),
        "max_time_delta_seconds": 3600,
    }
    r = client.post("/mosaics", json=p)
    assert r.status_code == 422
    assert "整数像元" in r.json()["detail"]


def test_png_encoding_roundtrip(client):
    """bands_png / cloud_mask_png 与数组编码等价。"""
    import base64
    import io

    from PIL import Image

    a = np.full((4, 4), 120, dtype=np.uint16)
    buf = io.BytesIO()
    Image.fromarray(a, mode="I;16").save(buf, format="PNG")
    bands_png = base64.b64encode(buf.getvalue()).decode()

    cm = np.zeros((4, 4), dtype=np.uint8)
    cm[0, 0] = 1
    buf2 = io.BytesIO()
    Image.fromarray(cm, mode="L").save(buf2, format="PNG")
    cloud_png = base64.b64encode(buf2.getvalue()).decode()

    tile_png = {
        "id": "png-tile",
        "acquired_at": "2026-09-01T10:00:00Z",
        "pixel_size": list(PIXEL),
        "origin": copy.deepcopy(ORIGIN),
        "nodata": -9999.0,
        "bands_png": bands_png,
        "bands_dtype": "uint16",
        "cloud_mask_png": cloud_png,
    }
    other = make_tile("arr-tile", "2026-09-02T10:00:00Z",
                      np.full((1, 4, 4), 999, np.int32))
    payload = {
        "tiles": [tile_png, other],
        "target": base_target(),
        "max_time_delta_seconds": 86400,
    }
    out = post(client, payload)
    bands = np.array(out["bands"])
    # 两时相清晰 -> 选新 arr-tile(999)，除 png tile 的云点 (0,0)
    assert bands[0, 0, 0] == 120
    assert bands[0, 1, 1] == 999
    assert out["preview_png"].startswith("data:image/png;base64,")
    assert out["source_map_png"].startswith("data:image/png;base64,")


def test_determinism(client):
    """相同语义（JSON 键顺序无关）的请求必须产生逐元素一致的结果。"""
    a = (np.arange(16).reshape(1, 4, 4) + 50).astype(np.int32)
    b = (np.arange(16).reshape(1, 4, 4) % 7 * 30 + 5).astype(np.int32)
    cloud = np.zeros((4, 4), np.uint8)
    cloud[1, 1] = 1
    payload = {
        "max_time_delta_seconds": 86400,
        "target": base_target(),
        "tiles": [
            make_tile("t1", "2026-09-01T10:00:00Z", a),
            make_tile("t2", "2026-09-02T10:00:00Z", b, cloud),
        ],
    }
    import json

    r1 = client.post("/mosaics", json=payload)
    reordered = json.loads(json.dumps(payload))
    r2 = client.post("/mosaics", json=reordered)
    j1, j2 = r1.json(), r2.json()
    assert j1["request_hash"] == j2["request_hash"]
    assert j1["id"] == j2["id"]  # 幂等命中
    assert j1["bands"] == j2["bands"]
    assert j1["source_map"] == j2["source_map"]
