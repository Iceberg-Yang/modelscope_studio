#!/usr/bin/env python3
"""GPU-free simulation for the ABot WebSocket playback policy.

The simulator models the parts that can be validated without loading the model:

* 12 decoded frames arrive together after each approximately one-second block;
* the server paces frames over a WebSocket with a bounded ACK window;
* ModelScope/browser round-trip time and modest network jitter;
* the browser's adaptive 10/11/12 FPS jitter buffer;
* control changes taking effect at the next sampled model block.

It is intentionally a transport-policy test, not a claim about GPU inference or
real Internet performance.  Run it before rebuilding the Docker image to reject
buffer settings that obviously trade latency for excessive underruns.
"""

from __future__ import annotations

import argparse
import heapq
import random
import statistics
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    name: str
    send_fps: float = 12.0
    catchup_fps: float = 12.0
    catchup_threshold: int = 1_000_000
    catchup_min_generated_fps: float = 0.0
    max_inflight: int = 3
    jitter_prime: int = 3
    jitter_target: int = 5
    jitter_max: int = 10
    slow_threshold: int = 3
    fast_threshold: int = 7
    display_fps_min: float = 10.0
    display_fps_target: float = 11.0
    display_fps_max: float = 12.0
    supply_aware_playback: bool = False


@dataclass(frozen=True)
class Frame:
    frame_id: int
    control_seq: int
    generated_at_ms: float
    generated_fps: float


@dataclass
class Result:
    display_age_p50_ms: float
    display_age_p95_ms: float
    control_response_p50_ms: float
    control_response_p95_ms: float
    underrun_rate_percent: float
    displayed_frames: int
    server_overflow_drops: int
    client_overflow_drops: int
    client_control_drops: int
    catchup_frames: int
    sent_frames: int


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def simulate(
    policy: Policy,
    *,
    seed: int,
    seconds: float,
    block_ms: float,
    block_jitter_ms: float,
    rtt_ms: float,
    network_jitter_ms: float,
) -> Result:
    scenario_rng = random.Random(seed)
    network_rng = random.Random(seed + 1_000_003)
    end_ms = int(seconds * 1000)

    controls: list[tuple[int, int]] = []
    control_at = 2200.0
    sequence = 0
    while control_at < end_ms - 1800:
        sequence += 1
        controls.append((int(control_at), sequence))
        control_at += scenario_rng.uniform(1800.0, 4200.0)

    server_queue: deque[Frame] = deque()
    inflight_acks: list[float] = []
    arrivals: list[tuple[float, int, Frame]] = []
    browser_buffer: list[Frame] = []
    control_sent_at: dict[int, float] = {}

    frame_id = 0
    arrival_order = 0
    requested_seq = 0
    block_seq = 0
    published_seq = 0
    active_seq = 0
    last_displayed_seq = 0
    control_index = 0

    def next_block_duration() -> float:
        return max(820.0, scenario_rng.gauss(block_ms, block_jitter_ms))

    block_started_at = 0.0
    next_block_at = next_block_duration()
    next_send_at = 0.0
    next_paint_tick = 0.0
    next_paint_at = 0.0
    paint_tick_ms = 1000.0 / 60.0
    playback_primed = False
    prime_threshold = policy.jitter_prime
    playback_started = False
    buffer_was_empty = True
    browser_supply_fps_ema = 0.0
    browser_last_generated_at = -1.0
    server_supply_fps_ema = 0.0
    server_last_generated_at = -1.0

    display_ages: list[float] = []
    control_responses: list[float] = []
    displayed_frames = 0
    underruns = 0
    server_overflow_drops = 0
    client_overflow_drops = 0
    client_control_drops = 0
    catchup_frames = 0
    sent_frames = 0

    for now in range(end_ms + 1):
        while control_index < len(controls) and controls[control_index][0] <= now:
            sent_at, requested_seq = controls[control_index]
            control_sent_at[requested_seq] = float(sent_at)
            control_index += 1

        while next_block_at <= now:
            generated_at = next_block_at
            generated_fps = 12_000.0 / max(1.0, next_block_at - block_started_at)
            if block_seq > published_seq:
                retained = deque(frame for frame in server_queue if frame.control_seq >= block_seq)
                server_queue = retained
                published_seq = block_seq
            for _ in range(12):
                frame_id += 1
                frame = Frame(
                    frame_id,
                    block_seq,
                    generated_at,
                    generated_fps,
                )
                if len(server_queue) >= 16:
                    server_queue.popleft()
                    server_overflow_drops += 1
                server_queue.append(frame)
            # Controls are sampled only when the following model block begins.
            block_seq = requested_seq
            block_started_at = next_block_at
            next_block_at += next_block_duration()

        while inflight_acks and inflight_acks[0] <= now:
            heapq.heappop(inflight_acks)

        if (
            server_queue
            and now >= next_send_at
            and len(inflight_acks) < policy.max_inflight
        ):
            frame = server_queue.popleft()
            if frame.generated_at_ms != server_last_generated_at:
                server_supply_fps_ema = (
                    0.25 * frame.generated_fps + 0.75 * server_supply_fps_ema
                    if server_supply_fps_ema
                    else frame.generated_fps
                )
                server_last_generated_at = frame.generated_at_ms
            one_way = max(
                1.0,
                network_rng.gauss(rtt_ms / 2.0, network_jitter_ms),
            )
            ack_delay = max(
                one_way + 1.0,
                network_rng.gauss(rtt_ms, network_jitter_ms * 1.5),
            )
            arrival_order += 1
            heapq.heappush(arrivals, (now + one_way, arrival_order, frame))
            heapq.heappush(inflight_acks, now + ack_delay)
            catchup = (
                len(server_queue) >= policy.catchup_threshold
                and server_supply_fps_ema >= policy.catchup_min_generated_fps
            )
            target_send_fps = policy.catchup_fps if catchup else policy.send_fps
            next_send_at = now + 1000.0 / target_send_fps
            sent_frames += 1
            if catchup:
                catchup_frames += 1

        while arrivals and arrivals[0][0] <= now:
            _, _, frame = heapq.heappop(arrivals)
            if frame.generated_at_ms != browser_last_generated_at:
                browser_supply_fps_ema = (
                    0.25 * frame.generated_fps + 0.75 * browser_supply_fps_ema
                    if browser_supply_fps_ema
                    else frame.generated_fps
                )
                browser_last_generated_at = frame.generated_at_ms
            if frame.control_seq < active_seq:
                continue
            if frame.control_seq >= requested_seq and frame.control_seq > active_seq:
                retained = [
                    item for item in browser_buffer
                    if item.control_seq >= frame.control_seq
                ]
                client_control_drops += len(browser_buffer) - len(retained)
                browser_buffer = retained
                active_seq = frame.control_seq
                playback_primed = False
                prime_threshold = 1
                next_paint_at = 0.0
            browser_buffer.append(frame)
            browser_buffer.sort(key=lambda item: item.frame_id)
            buffer_was_empty = False
            if not playback_primed and len(browser_buffer) >= prime_threshold:
                playback_primed = True
                playback_started = True
                prime_threshold = policy.jitter_prime
            if len(browser_buffer) > policy.jitter_max:
                while len(browser_buffer) > policy.jitter_target:
                    browser_buffer.pop(0)
                    client_overflow_drops += 1

        if now >= next_paint_tick:
            next_paint_tick += paint_tick_ms
            if not browser_buffer:
                if playback_started and not buffer_was_empty:
                    underruns += 1
                    buffer_was_empty = True
                playback_primed = False
                next_paint_at = 0.0
            elif playback_primed:
                slow_threshold = policy.slow_threshold
                if (
                    policy.supply_aware_playback
                    and 0 < browser_supply_fps_ema < policy.display_fps_target
                ):
                    slow_threshold = policy.jitter_target
                playback_fps = policy.display_fps_target
                if len(browser_buffer) <= slow_threshold:
                    playback_fps = policy.display_fps_min
                elif len(browser_buffer) >= policy.fast_threshold:
                    playback_fps = policy.display_fps_max
                frame_interval = 1000.0 / playback_fps
                if not next_paint_at:
                    next_paint_at = float(now)
                if now >= next_paint_at:
                    frame = browser_buffer.pop(0)
                    next_paint_at += frame_interval
                    if next_paint_at < now - frame_interval:
                        next_paint_at = now + frame_interval
                    display_ages.append(now - frame.generated_at_ms)
                    if frame.control_seq > last_displayed_seq:
                        candidates = [
                            sent_at
                            for seq, sent_at in control_sent_at.items()
                            if last_displayed_seq < seq <= frame.control_seq
                        ]
                        if candidates:
                            control_responses.append(now - min(candidates))
                        last_displayed_seq = frame.control_seq
                    displayed_frames += 1

    return Result(
        display_age_p50_ms=percentile(display_ages, 0.50),
        display_age_p95_ms=percentile(display_ages, 0.95),
        control_response_p50_ms=percentile(control_responses, 0.50),
        control_response_p95_ms=percentile(control_responses, 0.95),
        underrun_rate_percent=100.0 * underruns / max(1, displayed_frames),
        displayed_frames=displayed_frames,
        server_overflow_drops=server_overflow_drops,
        client_overflow_drops=client_overflow_drops,
        client_control_drops=client_control_drops,
        catchup_frames=catchup_frames,
        sent_frames=sent_frames,
    )


def aggregate(results: list[Result]) -> Result:
    def median(field: str) -> float:
        return float(statistics.median(getattr(result, field) for result in results))

    return Result(
        display_age_p50_ms=median("display_age_p50_ms"),
        display_age_p95_ms=median("display_age_p95_ms"),
        control_response_p50_ms=median("control_response_p50_ms"),
        control_response_p95_ms=median("control_response_p95_ms"),
        underrun_rate_percent=median("underrun_rate_percent"),
        displayed_frames=int(median("displayed_frames")),
        server_overflow_drops=int(median("server_overflow_drops")),
        client_overflow_drops=int(median("client_overflow_drops")),
        client_control_drops=int(median("client_control_drops")),
        catchup_frames=int(median("catchup_frames")),
        sent_frames=int(median("sent_frames")),
    )


def print_result(policy: Policy, result: Result) -> None:
    catchup_share = 100.0 * result.catchup_frames / max(1, result.sent_frames)
    print(f"\n{policy.name}")
    print(f"  display age       p50={result.display_age_p50_ms:7.1f}ms  p95={result.display_age_p95_ms:7.1f}ms")
    print(f"  control response  p50={result.control_response_p50_ms:7.1f}ms  p95={result.control_response_p95_ms:7.1f}ms")
    print(f"  underrun events       {result.underrun_rate_percent:7.3f}% of displayed frames")
    print(f"  displayed / drops     {result.displayed_frames} / server={result.server_overflow_drops}, client={result.client_overflow_drops}")
    print(f"  catch-up frames        {result.catchup_frames}/{result.sent_frames} ({catchup_share:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--block-ms", type=float, default=1012.0)
    parser.add_argument("--block-jitter-ms", type=float, default=45.0)
    parser.add_argument("--rtt-ms", type=float, default=189.0)
    parser.add_argument("--network-jitter-ms", type=float, default=12.0)
    parser.add_argument("--catchup-fps", type=float, default=14.0)
    parser.add_argument("--catchup-threshold", type=int, default=4)
    parser.add_argument("--catchup-min-generated-fps", type=float, default=11.5)
    parser.add_argument("--inflight", type=int, default=3)
    parser.add_argument("--prime", type=int, default=2)
    parser.add_argument("--target", type=int, default=3)
    parser.add_argument("--max-buffer", type=int, default=6)
    args = parser.parse_args()

    if not (1 <= args.prime <= args.target < args.max_buffer):
        parser.error("require 1 <= prime <= target < max-buffer")

    baseline = Policy(name="baseline (12 FPS sender, 3/5/10 buffer)")
    optimized = Policy(
        name=(
            f"optimized ({args.catchup_fps:g} FPS catch-up, "
            f"{args.prime}/{args.target}/{args.max_buffer} buffer)"
        ),
        catchup_fps=max(12.0, args.catchup_fps),
        catchup_threshold=max(1, args.catchup_threshold),
        catchup_min_generated_fps=max(0.0, args.catchup_min_generated_fps),
        max_inflight=max(1, args.inflight),
        jitter_prime=args.prime,
        jitter_target=args.target,
        jitter_max=args.max_buffer,
        slow_threshold=1,
        fast_threshold=args.target + 1,
        supply_aware_playback=True,
    )

    baseline_runs = []
    optimized_runs = []
    for seed in range(max(1, args.trials)):
        common = dict(
            seed=seed,
            seconds=max(15.0, args.seconds),
            block_ms=max(100.0, args.block_ms),
            block_jitter_ms=max(0.0, args.block_jitter_ms),
            rtt_ms=max(0.0, args.rtt_ms),
            network_jitter_ms=max(0.0, args.network_jitter_ms),
        )
        baseline_runs.append(simulate(baseline, **common))
        optimized_runs.append(simulate(optimized, **common))

    old = aggregate(baseline_runs)
    new = aggregate(optimized_runs)
    print(
        f"Scenario: {args.trials} trials x {args.seconds:.0f}s, "
        f"block={args.block_ms:.0f}ms, RTT={args.rtt_ms:.0f}ms"
    )
    print_result(baseline, old)
    print_result(optimized, new)

    display_gain = 100.0 * (1.0 - new.display_age_p50_ms / max(1.0, old.display_age_p50_ms))
    control_gain = 100.0 * (1.0 - new.control_response_p50_ms / max(1.0, old.control_response_p50_ms))
    print("\nComparison")
    print(f"  display p50 improvement: {display_gain:6.1f}%")
    print(f"  control p50 improvement: {control_gain:6.1f}%")

    # The simulator intentionally includes a block-boundary control delay.
    # Transport tuning must not promise to remove it; require no material
    # control regression and calibrate underruns against the simulated
    # baseline, whose absolute rate is slightly higher than the live metric.
    underrun_budget = max(3.0, old.underrun_rate_percent + 0.25)
    passed = (
        display_gain > 5.0
        and control_gain > -2.0
        and new.underrun_rate_percent <= underrun_budget
    )
    if passed:
        print("  verdict: PASS for Docker A/B validation")
        return 0
    print("  verdict: REVIEW settings before rebuilding")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
