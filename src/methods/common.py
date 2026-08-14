"""Shared helper for CoT/ToT/GoT: build the final output in exactly the
shape src/scorer.py expects, with a cost snapshot taken AFTER the method
has made all its LLM calls (so cost reflects only this run, not whatever
happened on the same lm instance before it -- callers must snapshot cost
before the run too and pass the delta; see methods/*.py for the pattern)."""


def build_output(state: dict, cost: dict) -> dict:
    return {
        "malicious_events": state.get("malicious_events", []),
        "narrative": state.get("narrative", ""),
        "_parse_ok": state.get("_parse_ok", True),
        "_cost": cost,
    }


def cost_delta(before: dict, after: dict) -> dict:
    return {
        "llm_calls": after["llm_calls"] - before["llm_calls"],
        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "total_tokens": after["total_tokens"] - before["total_tokens"],
        "wall_clock_s": round(after["wall_clock_s"] - before["wall_clock_s"], 3),
        "parse_retries": after["parse_retries"] - before["parse_retries"],
    }
