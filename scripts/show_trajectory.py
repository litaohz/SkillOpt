#!/usr/bin/env python3
"""Print an OfficeQA agent trajectory (tool calls) for one item + variant.

Reads ``<out_root>/<variant>/predictions/<UID>/conversation.json`` and prints
the turn-by-turn flow: user prompt, assistant messages, and each tool call with
its command + observation. Useful for reading what the agent actually did.

Example:
    python scripts/show_trajectory.py \
        --out-root outputs/attrib_ckpt_officeqa --variant empty --uid UID0032
"""
from __future__ import annotations

import argparse
import json
import os


def _s(v, n=300):
    if v is None:
        return ""
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s.strip()[:n] + ("  …" if len(s) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Show an OfficeQA agent trajectory")
    ap.add_argument("--out-root", required=True, help="e.g. outputs/attrib_ckpt_officeqa")
    ap.add_argument("--variant", default="full", help="full | empty | loo_NN | addone_NN")
    ap.add_argument("--uid", required=True, help="e.g. UID0032")
    ap.add_argument("--obs-chars", type=int, default=300)
    args = ap.parse_args()

    path = os.path.join(args.out_root, args.variant, "predictions", args.uid, "conversation.json")
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    conv = json.load(open(path, encoding="utf-8"))
    items = conv if isinstance(conv, list) else conv.get("messages", [])

    print(f"{'='*72}\n  {args.variant} / {args.uid}   ({len(items)} items)\n{'='*72}")
    for i, m in enumerate(items):
        typ = m.get("type")
        role = m.get("role")
        if typ == "tool_call" or "cmd" in m:
            print(f"[{i}] TOOL CALL: {_s(m.get('cmd'), 200)}")
            print(f"      -> obs: {_s(m.get('obs'), args.obs_chars)}")
        elif role == "user":
            print(f"[{i}] USER ({len(m.get('content') or '')} chars — question + oracle)")
        elif role == "assistant" or typ == "message":
            content = m.get("content")
            if content and str(content).strip():
                print(f"[{i}] ASSISTANT: {_s(content, 300)}")
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                print(f"      CALL {fn.get('name')}({_s(fn.get('arguments'), 200)})")
        elif role == "tool" or typ == "function_call_output":
            print(f"[{i}] TOOL RESULT: {_s(m.get('output') or m.get('content'), args.obs_chars)}")
        elif role == "system":
            print(f"[{i}] SYSTEM ({len(m.get('content') or '')} chars — rules + skill)")
        else:
            print(f"[{i}] {typ or role}: {_s(m.get('content'), 160)}")


if __name__ == "__main__":
    main()
