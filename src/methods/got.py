"""
Phase 4 -- Graph-of-Thoughts: split evidence into segments -> Generate per
segment independently -> Aggregate (merge) -> Refine. This is the method
the project's hypothesis is actually about: Aggregate is the one operation
CoT and ToT cannot express (a many-parent, one-child merge), and it's what
should let GoT connect events that look weak in isolation but become
obviously malicious once linked across segments.

Framework-integration note: graph_of_thoughts' Controller runs one shared
`problem_parameters` dict for an entire graph, which works cleanly for CoT
and ToT (one evidence_text for the whole run) but not for GoT's fan-out,
where each of N independent Generate calls needs a genuinely DIFFERENT
input (its own segment). So each segment's Generate step runs as its own
tiny Controller (sharing the same `lm` instance, so cost accumulates
correctly across all of them), and the Aggregate/Refine steps replicate
graph_of_thoughts' own Aggregate/Improve operation logic directly (compare
against operations.py's Aggregate._execute / Improve._execute -- same
prompter/parser calls, same state-merging) rather than fighting the
single-shared-state assumption. Every prompt template still comes from the
one shared KillChainPrompter used by CoT and ToT.
"""

from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations
from graph_of_thoughts.operations.operations import Generate
from graph_of_thoughts.controller.controller import Controller

from src.methods.common import build_output, cost_delta
from src.methods.parser import KillChainParser
from src.methods.prompter import KillChainPrompter


def split_into_segments(evidence_text: str, n_segments: int = 4) -> list[str]:
    """Split the serialized evidence into n_segments contiguous chunks of
    root-level process trees (root = a line with no leading indentation;
    everything indented beneath it belongs to that root), preserving the
    serializer's chronological root order.

    Balances by TEXT SIZE (characters), not by block COUNT. A real bug found
    by diagnosing WS12's complete GoT miss (Phase 4 smoke test): root blocks
    vary enormously in size (a handful of noisy processes, e.g. Chrome and
    Postgres, carry dozens of connected_to edges each), so splitting 82 root
    blocks into 4 equal-COUNT groups put all 6 ground-truth events plus
    ~78% of the entire evidence text into a single segment while the other
    three segments were nearly empty -- defeating the point of segmenting at
    all, since one "segment" was nearly the whole scenario. Balancing by
    character count instead spreads both the noise and the real evidence
    more evenly, so each segment gets an independent, genuinely partial view."""
    lines = evidence_text.split("\n")
    blocks = []
    current = []
    for line in lines:
        if line and not line[0].isspace():
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    if not blocks:
        return [evidence_text]

    block_texts = ["\n".join(b) for b in blocks]
    n_segments = max(1, min(n_segments, len(block_texts)))
    target_size = sum(len(b) for b in block_texts) / n_segments

    segments = []
    current_chunk = []
    current_size = 0
    for block_text in block_texts:
        current_chunk.append(block_text)
        current_size += len(block_text)
        # Close the current segment once it reaches its share of the total,
        # unless it's the last segment (which absorbs whatever remains) or
        # we've already produced all but one segment.
        if current_size >= target_size and len(segments) < n_segments - 1:
            segments.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0
    if current_chunk:
        segments.append("\n".join(current_chunk))
    return segments


def _generate_segment(segment_text: str, lm) -> dict:
    graph = GraphOfOperations()
    graph.append_operation(Generate(num_branches_prompt=1, num_branches_response=1))
    controller = Controller(
        lm=lm,
        graph=graph,
        prompter=KillChainPrompter(),
        parser=KillChainParser(lm=lm),
        problem_parameters={"evidence_text": segment_text},
    )
    controller.run()
    [thoughts] = controller.get_final_thoughts()
    return thoughts[0].state


def run_got(evidence_text: str, lm, n_segments: int = 4) -> dict:
    before = lm.cost_snapshot()
    prompter = KillChainPrompter()
    parser = KillChainParser(lm=lm)

    # Generate: reason over each segment independently, with no visibility
    # into the others -- this is what makes Aggregate's merge meaningful
    # rather than redundant with what a single call already saw.
    segments = split_into_segments(evidence_text, n_segments=n_segments)
    segment_states = [_generate_segment(seg, lm) for seg in segments]

    # Aggregate: merge the independent findings into one combined chain.
    # Mirrors graph_of_thoughts.operations.operations.Aggregate._execute.
    agg_prompt = prompter.aggregation_prompt(segment_states)
    [agg_text] = lm.get_response_texts(lm.query(agg_prompt, num_responses=1))
    merged_state = parser.parse_aggregation_answer(segment_states, [agg_text])

    # Refine: revise the merged answer once more with the full evidence in view.
    # Mirrors Improve._execute.
    refine_prompt = prompter.improve_prompt(evidence_text=evidence_text, **merged_state)
    [refine_text] = lm.get_response_texts(lm.query(refine_prompt, num_responses=1))
    final_state = parser.parse_improve_answer(merged_state, [refine_text])

    after = lm.cost_snapshot()
    return build_output(final_state, cost_delta(before, after))
