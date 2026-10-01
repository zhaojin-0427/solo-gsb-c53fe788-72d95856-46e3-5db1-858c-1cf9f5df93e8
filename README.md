# 多时相遥感云隙拼接 API (Multi-temporal Cloud-gap Mosaic API)

提交同一坐标网格下 **2～16 景** 带采集时间、像元分辨率、左上原点、波段矩阵、无数据值与云掩膜的栅格切片，服务按仿射位置落入输出网格，逐像元排除云和无数据，按**固定优先级**挑选来源像元，检测重叠冲突与无法覆盖的空洞，返回**确定性**拼接结果（矩阵 + 每像元来源 + 覆盖率 + 冲突证据），并将任务与输入摘要持久化到 SQLite，支持幂等重试。

技术栈：Python 3.11 · FastAPI · NumPy · Pillow · SQLite（零外部服务）。

---

## 1. 一键启动（Docker Compose）

```bash
docker compose up --build
```

启动后：

| 用途 | 地址 |
| --- | --- |
| API 服务 | <http://localhost:8000> |
| 健康检查 | <http://localhost:8000/health> |
| Swagger 交互文档 | <http://localhost:8000/docs> |
| ReDoc 文档 | <http://localhost:8000/redoc> |
| OpenAPI Schema | <http://localhost:8000/openapi.json> |

SQLite 数据库持久化在命名卷 `mosaic-data:/data/mosaics.db`，容器重启后历史任务仍可查询。

### 本地直接运行（可选）

```bash
python -m pip install -r requirements.txt
export MOSAIC_DB_PATH=./data/mosaics.db
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 2. 接口

### `POST /mosaics` — 创建拼接任务（同步计算并持久化）

请求体核心字段：

```jsonc
{
  "target": {                       // 输出网格
    "origin_x": 0.0, "origin_y": 80.0,   // 输出网格左上角的坐标
    "pixel_size": 10.0,                  // 每像元地面分辨率（>0）
    "width": 8, "height": 8
  },
  "tiles": [                        // 2～16 景切片
    {
      "tile_id": "S2A-20260828",
      "acquired_at": "2026-08-28T10:00:00Z",  // ISO-8601 采集时间
      "pixel_size": 10.0,
      "origin_x": 0.0, "origin_y": 80.0,
      "bands": [ [[...]], [[...]] ],         // [band][row][col]，各景波段数一致
      "nodata": -1.0,                        // 无数据值（可选）
      "cloud_mask": [[0, 1, ...]]            // HxW 的 0/1 云掩膜（可选，默认全晴）
    }
  ],
  "reference_time": "2026-08-31T00:00:00Z",  // 时间接近度基准（可选，默认取各景时间中位数）
  "max_time_delta_seconds": 259200,          // 时间窗：超出的景不参与选源
  "conflict_tolerance": 25.0,                // 重叠像元值差超过该值记为冲突
  "max_evidence": 200,                       // 冲突/空洞证据最多返回的条数
  "request_id": "demo-2026-w36"              // 客户端幂等键（可选）
}
```

坐标系约定：北朝上仿射，`X = origin_x + col * pixel_size`，`Y = origin_y - row * pixel_size`（行向南递增）。服务不绑定 CRS，所有坐标只需处于同一参考系。

**网格对齐规则**（不满足返回 `422`）：

* 所有切片原点必须落在输出网格的像元角点上；
* 切片分辨率是输出的**整数倍**（更粗 → 最近邻复制填充），或输出是切片的**整数倍**（更细 → 对晴空有效子像元取均值，输入摘要中 `mode` 分别为 `replicate_coarser` / `aggregate_finer` / `native`）。

### `GET /mosaics/{id}` — 查询历史结果

`{id}` 可以是数字主键（`1`）或提交时的 `request_id`（`demo-2026-w36`）。

### `GET /mosaics/{id}/preview.png` — 结果预览图

Pillow 渲染：第 1 波段灰度线性拉伸，空洞着红色，便于快速目检。

### `GET /health`

### 幂等与冲突

* 相同 `request_id` + 相同请求体重放：直接返回已存结果，响应中 `"reused": true`，不产生新主键；
* 相同 `request_id` 但**不同**请求体：返回 `409 Conflict`；
* 即使不带 `request_id`，每个结果也会持久化，可用数字 id 复核。

---

## 3. 选源优先级（固定、确定性）

对每个输出像元，候选 = 覆盖该像元、非云、至少一个波段有效、且在时间窗内的切片。按以下固定顺序比较，先决胜先得：

1. **切片云量**（整景云占比，越低越优先）——优先使用最晴的一景；
2. **时间接近度** `|acquired_at - reference_time|`（越小越优先）；
3. **离切片覆盖边缘的距离**（精确欧氏距离变换，单位为地面单位，离接缝越远越优先）；
4. **`tile_id` 字典序**（最终确定性平局裁决）。

因此：输入切片顺序不同、重复提交、跨语言 JSON 键序不同，结果矩阵逐像元一致。更细分辨率切片若子像元全部有云则计为云；部分有云时按晴空子像元均值聚合，云占比作为次级云信息保留在诊断中。

---

## 4. 返回结果

```jsonc
{
  "id": 1, "status": "completed", "request_id": "demo-2026-w36",
  "reference_time": "2026-08-31T00:00:00Z",
  "mosaic": [ /* [band][row][col]，空洞为 null */ ],
  "sources": [ /* [row][col] 获胜 tile_id，空洞为 null */ ],
  "selected_index": [ /* [row][col] 1-based 输入序号，0 = 空洞 */ ],
  "hole_mask": [ /* 1 = 无法覆盖 */ ],
  "hole_reasons": [ /* no_coverage | cloud | nodata | outside_time_window */ ],
  "coverage_fraction": 1.0,           // 总体覆盖率
  "band_coverage_fraction": [1.0],    // 分波段有效率
  "conflict_count": 0,                // 存在 ≥2 个有效值且极差 > tolerance 的像元数
  "conflict_fraction": 0.0,
  "conflict_evidence": [              // 每个冲突像元：行列、波段、来源与数值证据
    {"row": 0, "col": 5, "band": 0,
     "sources": ["S2A-20260828", "S2B-20260830"],
     "values": [505.0, 605.0], "difference": 100.0}
  ],
  "holes": [ {"row": 0, "col": 0, "reason": "cloud"} ],
  "warnings": ["..."],                // 例如时间窗排除的景、细分辨率边缘未对齐提示
  "input_summary": [ /* 每景形状、云量、覆盖像元数、重采样模式 */ ],
  "created_at": "2026-10-01T12:00:00Z",
  "reused": false
}
```

空洞原因按证据优先级归类：`no_coverage`（无任何切片覆盖）→ `cloud`（覆盖切片全为云）→ `nodata`（覆盖且晴空但全为无数据）→ `outside_time_window`（唯一有效来源在时间窗外）。

---

## 5. 直接调用复核

仓库自带示例 `examples/request.json`（3 景互补云洞的 8×8 场景）：

```bash
# 创建
curl -sS -X POST http://localhost:8000/mosaics \
  -H 'Content-Type: application/json' \
  --data @examples/request.json | python -m json.tool | less

# 幂等重试（同一 request_id，返回 reused: true，id 不变）
curl -sS -X POST http://localhost:8000/mosaics \
  -H 'Content-Type: application/json' \
  --data @examples/request.json | python -c "import sys,json;r=json.load(sys.stdin);print(r['id'], r['reused'], r['coverage_fraction'])"

# 按 request_id 复核
curl -sS http://localhost:8000/mosaics/demo-2026-w36 | python -m json.tool | less

# 按数字主键复核
curl -sS http://localhost:8000/mosaics/1 | python -m json.tool | less

# 下载预览图（空洞红色）
curl -sS http://localhost:8000/mosaics/1/preview.png -o preview.png && xdg-open preview.png
```

浏览器打开 <http://localhost:8000/docs> 可直接在 Swagger UI 中粘贴 `examples/request.json` 发送请求并浏览完整响应。

---

## 6. 测试

```bash
python -m pip install pytest httpx
python -m pytest tests/ -q
```

测试覆盖：云隙互补填充、云量/时间/边缘优先级、粗细分辨率重采样、冲突检测与容差、四类空洞归因、确定性（打乱输入结果一致）、`POST/GET` 往返、幂等重试与 `409`、参数校验 `422`、PNG 预览，以及距离变换对暴力枚举结果的逐点精确校验。

## 7. 限制

* 单景与输出网格边长上限 2048 像元，波段数 ≤ 8，切片数 2～16（可在 `app/schemas.py` 调整）；
* 多波段时以“至少一个波段有效”作为像元参与条件，冲突逐波段独立判定；
* 同步处理，超大场景请先切分窗口（API 本身按窗口设计）。
