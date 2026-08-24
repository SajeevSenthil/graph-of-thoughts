"""
Phase 4 -- Tree-of-Thoughts: Generate k candidate reconstructions -> Evaluate
(score each) -> KeepBest. Uses the exact same generate_prompt as CoT (see
prompter.py's module docstring for why) -- the only difference is asking for
k sampled branches instead of 1, then explicitly scoring and pruning them,
which is what gives ToT the ability to reconsider an early wrong turn that
CoT structurally cannot.

Scoring uses graph_of_thoughts' `scoring_function` callable path (see
Score._execute: when scoring_function is set, it's called directly on each
thought's state, bypassing prompter.score_prompt/parser.parse_score_answer
entirely) rather than a second prompt template, since "how plausible is this
whole candidate" is naturally a single free-form question, not a distinct
templated prompt shape worth duplicating.
"""

import re

from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations
from graph_of_thoughts.operations.operations import Generate, KeepBestN, Score
from graph_of_thoughts.controller.controller import Controller

from src.methods.common import build_output, cost_delta
from src.methods.parser import KillChainParser
from src.methods.prompter import KillChainPrompter

_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _make_scoring_function(lm, evidence_text: str):
    def score_candidate(state: dict) -> float:
        import json

        reconstruction = json.dumps(
            {"malicious_events": state.get("malicious_events", []), "narrative": state.get("narrative", "")}
        )
        prompt = (
            "Rate how well this attack reconstruction is supported by the evidence below, "
            "on a scale from 0.0 (not supported / likely wrong) to 1.0 (well supported / correct).\n\n"
            f"Evidence:\n{evidence_text}\n\nReconstruction:\n{reconstruction}\n\n"
            "Respond with ONLY a single number between 0.0 and 1.0, nothing else."
        )
        # structured=False: this prompt asks for a single confidence float,
        # not the {malicious_events, narrative} schema llm_client.py forces
        # by default -- see query()'s docstring note.
        [text] = lm.get_response_texts(lm.query(prompt, num_responses=1, structured=False))
        match = _SCORE_RE.search(text)
        if not match:
            return 0.0
        return max(0.0, min(1.0, float(match.group(1))))

    return score_candidate


def run_tot(evidence_text: str, lm, k: int = 3) -> dict:
    before = lm.cost_snapshot()

    graph = GraphOfOperations()
    graph.append_operation(Generate(num_branches_prompt=1, num_branches_response=k))
    graph.append_operation(
        Score(num_samples=1, combined_scoring=False, scoring_function=_make_scoring_function(lm, evidence_text))
    )
    graph.append_operation(KeepBestN(n=1, higher_is_better=True))

    controller = Controller(
        lm=lm,
        graph=graph,
        prompter=KillChainPrompter(),
        parser=KillChainParser(lm=lm),
        problem_parameters={"evidence_text": evidence_text},
    )
    controller.run()

    [thoughts] = controller.get_final_thoughts()
    final_state = thoughts[0].state

    after = lm.cost_snapshot()
    return build_output(final_state, cost_delta(before, after))
