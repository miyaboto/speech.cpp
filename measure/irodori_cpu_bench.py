#!/usr/bin/env python3
# Reproduces the CPU-only Irodori-TTS timing table: stage times, first audio, total and RTF for fixed Japanese
# sentences, and optionally a CER against Qwen3-ASR or FastConformer. Needs a built irodori-timing (and speech when
# --asr is given).
#
# usage:
#   python3 measure/irodori_cpu_bench.py --speech build/speech --timing build/irodori-timing \
#       --model models/Irodori-TTS-866M-MF-v4.1-F16.gguf [--threads N] [--steps N] [--runs 3] \
#       [--asr qwen3-asr-0.6b|parakeet-tdt_ctc-0.6b-ja] [--out-dir /tmp/irodori-cpu-bench]

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SENTENCES = [
    "もしもし、宮本様のお電話でお間違いないでしょうか。",
    "ありがとうございます。本日はご予約の確認でお電話いたしました。",
]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def cpu_label() -> str:
    model = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    cores = os.cpu_count() or 0
    return f"{model}, {cores} logical CPUs, {platform.system()} {platform.release()}"


def parse_timing_lines(stdout: str) -> list[dict]:
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def cer(ref: str, hyp: str) -> float:
    """Character error rate after stripping spaces and punctuation common in ASR output."""
    def norm(s: str) -> str:
        s = s.strip()
        s = re.sub(r"[\s　、。．，,.!?！？「」『』（）()【】\[\]…・]", "", s)
        return s

    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # Wagner–Fischer
    dp = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, dp[0] = dp[0], i
        for j, hc in enumerate(h, 1):
            cur = dp[j]
            dp[j] = prev if rc == hc else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(r)


def synthesize_wav(speech: Path, model: str, text: str, out: Path, threads: int | None, steps: int | None, speed: float, seed: int) -> None:
    cmd = [str(speech), "tts", model, "--device", "cpu"]
    if threads:
        cmd += ["--threads", str(threads)]
    cmd += ["--voice", "none", "--speed", str(speed), "--seed", str(seed)]
    if steps:
        cmd += ["--steps", str(steps)]
    cmd += ["-o", str(out), text]
    run(cmd)


def asr_text(speech: Path, asr_model: str, wav: Path) -> str:
    proc = run([str(speech), "asr", asr_model, "--device", "cpu", str(wav)])
    # speech asr prints the transcript on stdout.
    return proc.stdout.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speech", type=Path, help="path to speech (needed for --asr and for writing WAVs)")
    p.add_argument("--timing", type=Path, required=True, help="path to irodori-timing")
    p.add_argument("--model", required=True, help="Irodori-TTS GGUF path or catalog name")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--asr", default="", help="recognition model for CER, e.g. qwen3-asr-0.6b")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if not args.timing.is_file():
        print(f"error: irodori-timing not found at {args.timing}", file=sys.stderr)
        return 2
    if args.asr and (not args.speech or not args.speech.is_file()):
        print("error: --asr needs --speech pointing at a built speech binary", file=sys.stderr)
        return 2

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="irodori-cpu-bench-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(args.timing), args.model, "cpu", str(args.runs), "--speed", str(args.speed), "--seed", str(args.seed)]
    if args.threads:
        cmd.extend(["--threads", str(args.threads)])
    if args.steps:
        cmd.extend(["--steps", str(args.steps)])
    cmd.append("--")
    cmd.extend(SENTENCES)
    print(f"hardware: {cpu_label()}", file=sys.stderr)
    print(f"command: {' '.join(cmd)}", file=sys.stderr)
    proc = run(cmd)
    rows = parse_timing_lines(proc.stdout)
    if len(rows) != len(SENTENCES):
        print(proc.stdout, file=sys.stderr)
        print(f"error: expected {len(SENTENCES)} timing lines, got {len(rows)}", file=sys.stderr)
        return 1

    print("| sentence | first audio | total | RTF | text | sampling | codec | CER |")
    print("|---|---|---|---|---|---|---|---|")
    for text, row in zip(SENTENCES, rows):
        cer_s = ""
        if args.asr:
            wav = out_dir / f"seed{args.seed}_steps{row['steps']}.wav"
            # One WAV per sentence.
            wav = out_dir / (re.sub(r"\W+", "_", text)[:40] + ".wav")
            synthesize_wav(args.speech, args.model, text, wav, args.threads or None, args.steps or None, args.speed, args.seed)
            hyp = asr_text(args.speech, args.asr, wav)
            (out_dir / (wav.stem + ".txt")).write_text(hyp + "\n", encoding="utf-8")
            cer_s = f"{cer(text, hyp) * 100:.1f}%"
            print(f"  asr: {hyp!r}", file=sys.stderr)
        short = text if len(text) <= 24 else text[:24] + "…"
        print(
            f"| {short} | {row['first_audio']:.3f} s | {row['total']:.3f} s | {row['rtf']:.3f} | "
            f"{row['text_s']:.3f} s | {row['sampling_s']:.3f} s | {row['codec_s']:.3f} s | {cer_s or '—'} |"
        )

    summary = {
        "hardware": cpu_label(),
        "model": args.model,
        "threads": rows[0]["threads"],
        "steps": rows[0]["steps"],
        "speed": args.speed,
        "seed": args.seed,
        "runs": args.runs,
        "rows": rows,
    }
    (out_dir / "timing.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'timing.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
