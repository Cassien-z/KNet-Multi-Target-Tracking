"""Black-box functional acceptance checks for the prototype backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import websockets


class AcceptanceFailure(RuntimeError):
    pass


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[Any, str]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get_content_type()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AcceptanceFailure(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise AcceptanceFailure(f"无法连接后端 {base_url}: {exc.reason}") from exc
    if content_type == "application/json":
        return json.loads(raw), content_type
    return raw, content_type


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)
    print(f"[PASS] {message}")


async def _check_websocket(base_url: str, scenario_id: str) -> None:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"dataset": scenario_id, "target": 0})
    uri = urlunparse((scheme, parsed.netloc, "/ws/tracking", "", query, ""))
    async with websockets.connect(uri, open_timeout=30, close_timeout=5) as socket:
        frame = json.loads(await asyncio.wait_for(socket.recv(), timeout=120))
        await socket.send(json.dumps({"action": "pause"}))
        await asyncio.sleep(0.15)
        seek_time = min(5.0, float(frame.get("max_time", 5.0)))
        await socket.send(json.dumps({"action": "seek", "time": seek_time}))
        await socket.send(json.dumps({"action": "play"}))
        seek_frame = None
        for _ in range(20):
            candidate = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
            if candidate.get("is_reset") and float(candidate.get("time", 0)) >= seek_time:
                seek_frame = candidate
                break
    required = {"time", "truth_pos", "pred_pos", "metrics"}
    _check(required.issubset(frame), "WebSocket 能返回真实跟踪帧")
    _check(len(frame["truth_pos"]) == 3 and len(frame["pred_pos"]) == 3, "跟踪帧三维位置结构正确")
    _check(seek_frame is not None, "WebSocket 支持暂停、跳转和继续播放")


def run(args: argparse.Namespace) -> int:
    health, _ = _request(args.base_url, "/api/health")
    _check(health.get("status") == "ready", "健康检查状态为 ready")
    _check(health.get("active_model", {}).get("model") == "CoopTrackerModel", "已部署 CoopTrackerModel")

    system, _ = _request(args.base_url, "/api/system")
    _check(system.get("runtime", {}).get("pytorch"), "系统信息包含 PyTorch 运行时")

    models, _ = _request(args.base_url, "/api/models")
    ids = {item["id"] for item in models.get("items", [])}
    _check(ids == {"v5", "v6", "v7"}, "模型仓库仅注册 V5、V6、V7 三个最佳权重")

    activation, _ = _request(
        args.base_url,
        "/api/models/activate",
        method="POST",
        payload={"model_id": args.model, "device": args.device},
        timeout=120,
    )
    _check(activation.get("model_id") == args.model, f"模型 {args.model} 可激活")

    datasets, _ = _request(args.base_url, "/api/datasets")
    dataset_map = {item["id"]: item for item in datasets.get("items", [])}
    _check(args.dataset in dataset_map and dataset_map[args.dataset]["available"], f"数据集 {args.dataset} 可用")

    criteria, _ = _request(args.base_url, "/api/config/acceptance-criteria")
    _check(criteria.get("total") == 4, "任务书四项技术指标已配置")

    scenarios, _ = _request(args.base_url, "/api/scenarios?limit=1")
    _check(bool(scenarios.get("items")), "场景目录可读取")
    asyncio.run(_check_websocket(args.base_url, str(scenarios["items"][0]["id"])))

    references, _ = _request(args.base_url, "/api/evaluations/references")
    _check(references.get("imm_jpda_comparison") is not None, "IMM-JPDA 历史对比可查询")
    matrix, _ = _request(args.base_url, "/api/evaluations/cross-dataset")
    _check(matrix.get("available") and len(matrix.get("matrix", [])) == 12, "3×4 跨数据集结果可查询")
    comparison, _ = _request(args.base_url, "/api/evaluations/compare")
    comparison_rows = comparison.get("items", [])
    _check(
        all(row.get("model_id") and row.get("dataset_id") for row in comparison_rows),
        "历史批测对比包含模型与数据集标识",
    )
    reports, _ = _request(args.base_url, "/api/reports")
    _check("summary" in reports and "items" in reports, "报告中心归档目录可查询")
    completed_reports = [item for item in reports.get("items", []) if item.get("status") == "completed"]
    if completed_reports:
        report_id = completed_reports[0]["id"]
        bundle, bundle_type = _request(
            args.base_url, f"/api/reports/{report_id}/assessment.json"
        )
        _check(
            bundle_type == "application/json"
            and bundle.get("kind") == "tracking_batch_assessment",
            "综合评估 JSON 可下载",
        )
        selected_ids = ",".join(item["id"] for item in completed_reports[:2])
        comparison_csv, comparison_type = _request(
            args.base_url, f"/api/reports/comparison.csv?job_ids={selected_ids}"
        )
        _check(
            comparison_type == "text/csv" and b"position_rmse_m" in comparison_csv,
            "多任务对比 CSV 可下载",
        )

    if args.skip_batch:
        print("[SKIP] 已按参数跳过真实批量测试")
        return 0

    job, _ = _request(
        args.base_url,
        "/api/batch-tests",
        method="POST",
        payload={
            "name": "后端自动验收",
            "model_id": args.model,
            "dataset_id": args.dataset,
            "device": args.device,
            "warmup_frames": 5,
            "batch_size": 64,
            "workers": 0,
            "max_scenarios": args.max_scenarios,
        },
    )
    job_id = job["id"]
    print(f"[INFO] 批测任务 {job_id} 已提交")
    deadline = time.monotonic() + args.batch_timeout
    while time.monotonic() < deadline:
        job, _ = _request(args.base_url, f"/api/batch-tests/{job_id}")
        status = job.get("status")
        progress = job.get("progress", {})
        print(f"[INFO] batch={status} progress={progress.get('percent', 0)}%", end="\r")
        if status in {"completed", "failed", "cancelled"}:
            print()
            break
        time.sleep(1.0)
    else:
        raise AcceptanceFailure(f"批测任务在 {args.batch_timeout:.0f}s 内未结束")
    _check(job.get("status") == "completed", f"真实批量测试完成（job_id={job_id}）")
    _check(job.get("result", {}).get("metrics", {}).get("evaluated_tracks", 0) > 0, "批测产生有效精度指标")

    assessment, _ = _request(args.base_url, f"/api/batch-tests/{job_id}/assessment")
    _check(len(assessment.get("criteria", [])) == 4, "批测结果可生成多维指标判定")
    print(f"[INFO] 当前小样本指标结论：{assessment['overall_status']}（数据不足不等同于不达标）")

    report, report_type = _request(args.base_url, f"/api/batch-tests/{job_id}/report.json")
    _check(report_type == "application/json" and report.get("id") == job_id, "JSON 报告可下载")
    csv_body, csv_type = _request(args.base_url, f"/api/batch-tests/{job_id}/tracks.csv")
    _check(csv_type == "text/csv" and b"rmse_m" in csv_body, "逐航迹 CSV 可下载")

    print(f"\n后端功能验收通过。验收批测 job_id={job_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端融合跟踪后端自动验收")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", choices=("v5", "v6", "v7"), default="v7")
    parser.add_argument("--dataset", choices=("v5", "v6", "v7", "v8_interference"), default="v7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-scenarios", type=int, default=1)
    parser.add_argument("--batch-timeout", type=float, default=600.0)
    parser.add_argument("--skip-batch", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (AcceptanceFailure, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
