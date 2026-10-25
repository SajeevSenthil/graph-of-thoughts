"""
Phase 4 -- Parser implementation shared by CoT, ToT, and GoT. Same reasoning
as prompter.py: identical parsing logic for every topology, so nothing about
how output is interpreted can differ by condition.
"""

import json
import re
from typing import Dict, List

from graph_of_thoughts.parser.parser import Parser

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# Reasoning models (e.g. Llama-Primus-Reasoning) emit a <think>...</think>
# block before their actual JSON output. Strip it before any JSON search so
# a {…} inside the think block can never be mistaken for the answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

EMPTY_RECONSTRUCTION = {"malicious_events": [], "narrative": ""}


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning traces. Also handles the case where
    the model stopped generating mid-thought (no closing tag)."""
    stripped = _THINK_RE.sub("", text).strip()
    # Unclosed <think> tag: drop everything from it onward (the model never
    # finished reasoning and there is no answer text after it).
    lower = stripped.lower()
    if "<think>" in lower:
        stripped = stripped[: lower.index("<think>")].strip()
    return stripped


def extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from a model response: strips reasoning
    traces and markdown code fences if present, then takes the largest {...}
    block and parses it.  Returns None (never raises) if nothing parseable is
    found -- callers treat that as a failed attempt and count a retry."""
    text = _strip_thinking(text)
    fenced = _CODE_FENCE_RE.search(text)
    candidate_text = fenced.group(1) if fenced else text

    match = _JSON_BLOCK_RE.search(candidate_text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or "malicious_events" not in parsed:
        return None
    if not isinstance(parsed.get("malicious_events"), list):
        return None
    return {
        "malicious_events": parsed["malicious_events"],
        "narrative": parsed.get("narrative", ""),
    }


class KillChainParser(Parser):
    def __init__(self, lm=None):
        # lm is optional and only used to increment lm.parse_retries when a
        # response fails to parse -- purely for the cost/robustness metrics
        # described to the user, not for correctness.
        self.lm = lm

    def _note_retry(self):
        if self.lm is not None:
            self.lm.parse_retries += 1

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            parsed = extract_json(text)
            if parsed is None:
                self._note_retry()
                results.append({**EMPTY_RECONSTRUCTION, "_parse_ok": False, "_raw_text": text})
            else:
                results.append({**parsed, "_parse_ok": True})
        return results

    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> Dict:
        parsed = extract_json(texts[0]) if texts else None
        if parsed is None:
            self._note_retry()
            return {**EMPTY_RECONSTRUCTION, "_parse_ok": False, "_raw_text": texts[0] if texts else ""}
        return {**parsed, "_parse_ok": True}

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        parsed = extract_json(texts[0]) if texts else None
        if parsed is None:
            self._note_retry()
            return {**state, "_parse_ok": False}  # keep prior state rather than discarding it
        return {**parsed, "_parse_ok": True}

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        raise NotImplementedError("validation_prompt/parse_validation_answer are not used by this project")

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        raise NotImplementedError("parse_score_answer is not used -- ToT scores via a scoring_function callable")
