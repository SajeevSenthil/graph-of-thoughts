"""
Phase 4 -- Chain-of-Thought: one LLM call, linear reasoning, no revisiting.
The baseline every other topology is compared against.
"""

from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations
from graph_of_thoughts.operations.operations import Generate
from graph_of_thoughts.controller.controller import Controller

from src.methods.common import build_output, cost_delta
from src.methods.parser import KillChainParser
from src.methods.prompter import KillChainPrompter


def run_cot(evidence_text: str, lm) -> dict:
    before = lm.cost_snapshot()

    graph = GraphOfOperations()
    graph.append_operation(Generate(num_branches_prompt=1, num_branches_response=1))

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
