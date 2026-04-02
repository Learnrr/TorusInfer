#!/usr/bin/env python3
"""
Simple concurrent benchmark for /v1/completions endpoint.

Examples:
  python benchmark/benchmark_concurrency.py \
    --base-url http://127.0.0.1:8000 \
    --prompt "Write a short poem." \
    --requests 50 \
    --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class RequestResult:
    ok: bool
    latency_ms: float
    status_code: int
    response_size: int
    error: str = ""


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)

    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_vals[low]
    frac = rank - low
    return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac


def build_payload(
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    presence_penalty: float,
    frequency_penalty: float,
) -> Dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }


def send_one_request(url: str, payload: Dict[str, Any], timeout_s: float) -> RequestResult:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            content = resp.read()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            status = int(getattr(resp, "status", 200))
            ok = 200 <= status < 300
            return RequestResult(
                ok=ok,
                latency_ms=elapsed_ms,
                status_code=status,
                response_size=len(content),
                error="" if ok else f"HTTP {status}",
            )
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            body_bytes = e.read()
            details = body_bytes.decode("utf-8", errors="ignore")[:200]
        except Exception:
            details = ""
        msg = f"HTTPError {e.code}"
        if details:
            msg += f": {details}"
        return RequestResult(
            ok=False,
            latency_ms=elapsed_ms,
            status_code=int(e.code),
            response_size=0,
            error=msg,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(
            ok=False,
            latency_ms=elapsed_ms,
            status_code=0,
            response_size=0,
            error=str(e),
        )


def run_benchmark(
    url: str,
    payload: Dict[str, Any],
    total_requests: int,
    concurrency: int,
    timeout_s: float,
) -> Tuple[List[RequestResult], float]:
    results: List[RequestResult] = []
    lock = threading.Lock()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_one_request, url, payload, timeout_s)
            for _ in range(total_requests)
        ]
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results.append(r)

    elapsed_s = time.perf_counter() - started
    return results, elapsed_s


def print_report(results: List[RequestResult], elapsed_s: float) -> None:
    total = len(results)
    ok_count = sum(1 for r in results if r.ok)
    fail_count = total - ok_count

    lat_all = [r.latency_ms for r in results]
    lat_ok = [r.latency_ms for r in results if r.ok]

    rps = total / elapsed_s if elapsed_s > 0 else float("nan")
    ok_rps = ok_count / elapsed_s if elapsed_s > 0 else float("nan")

    print("\n=== Benchmark Report ===")
    print(f"Total requests:      {total}")
    print(f"Success:             {ok_count}")
    print(f"Failed:              {fail_count}")
    print(f"Wall time (s):       {elapsed_s:.3f}")
    print(f"Throughput req/s:    {rps:.2f}")
    print(f"Success req/s:       {ok_rps:.2f}")

    if lat_all:
        print("\nLatency (all, ms):")
        print(f"  min:               {min(lat_all):.2f}")
        print(f"  mean:              {statistics.mean(lat_all):.2f}")
        print(f"  p50:               {percentile(lat_all, 50):.2f}")
        print(f"  p95:               {percentile(lat_all, 95):.2f}")
        print(f"  p99:               {percentile(lat_all, 99):.2f}")
        print(f"  max:               {max(lat_all):.2f}")

    if fail_count > 0:
        print("\nSample errors:")
        shown = 0
        for r in results:
            if not r.ok:
                print(f"  - {r.error}")
                shown += 1
                if shown >= 5:
                    break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent benchmark for infer2 /v1/completions")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--model", default="QWEN", help="Model name sent in request body")
    parser.add_argument("--prompt", default="Explain KV cache in one paragraph.", help="Prompt text")

    parser.add_argument("--requests", type=int, default=100, help="Total request count")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent workers")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)

    parser.add_argument("--warmup", type=int, default=0, help="Warmup requests before benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.requests <= 0:
        raise ValueError("--requests must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    url = args.base_url.rstrip("/") + "/v1/completions"

    payload = build_payload(
        model=args.model,
        prompt=args.prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
    )

    print("=== Benchmark Config ===")
    print(f"URL:                {url}")
    print(f"Total requests:     {args.requests}")
    print(f"Concurrency:        {args.concurrency}")
    print(f"Timeout per req(s): {args.timeout}")

    if args.warmup > 0:
        print(f"\nRunning warmup: {args.warmup} request(s)...")
        for _ in range(args.warmup):
            _ = send_one_request(url, payload, args.timeout)

    results, elapsed_s = run_benchmark(
        url=url,
        payload=payload,
        total_requests=args.requests,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
    )
    print_report(results, elapsed_s)


if __name__ == "__main__":
    main()
