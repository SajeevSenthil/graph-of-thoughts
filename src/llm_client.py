"""
Phase 4 -- vLLM wrapper implementing spcl/graph-of-thoughts' AbstractLanguageModel
interface, so CoT/ToT/GoT all go through the same LM plumbing (query +
get_response_texts) regardless of topology.

Bakes in the environment fixes discovered during Phase 0 (see CLAUDE.md
"Working conventions"), applied here so they hold no matter how this module
is invoked (venv activated or not, script or notebook):
  - VLLM_CACHE_ROOT / TRITON_CACHE_DIR pointed at local disk, not the
    GlusterFS-backed home directory -- Triton's JIT compile cache breaks
    there (errno 61: No data available) writing temp files during autotuning.
  - `.venv/bin` (sys.exec_prefix/bin) prepended to PATH -- FlashInfer's JIT
    kernel build shells out to `ninja`, which is only found via PATH, not via
    the interpreter path alone.
Must happen BEFORE `import vllm`, since vLLM reads some of these at import time.
"""

import os
import sys

_LOCAL_CACHE_ROOT = "/var/tmp/c_prajisha_cache"
os.environ.setdefault("VLLM_CACHE_ROOT", f"{_LOCAL_CACHE_ROOT}/vllm_compile")
os.environ.setdefault("TRITON_CACHE_DIR", f"{_LOCAL_CACHE_ROOT}/triton")
_venv_bin = os.path.join(sys.exec_prefix, "bin")
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

import time  # noqa: E402
import types  # noqa: E402
from typing import List  # noqa: E402

# graph_of_thoughts==0.0.2's controller/__init__.py unconditionally imports its
# ChatGPT class, which references `openai.error.OpenAIError` -- an attribute
# that only existed in openai<1.0. Our installed openai SDK is newer and has
# no `.error` submodule at all, so any import from graph_of_thoughts.controller
# (even just for AbstractLanguageModel, which has nothing to do with OpenAI)
# fails at import time. We never instantiate ChatGPT, so stub the missing
# attribute rather than downgrading openai or patching site-packages.
import openai  # noqa: E402

if not hasattr(openai, "error"):
    openai.error = types.ModuleType("openai.error")
    openai.error.OpenAIError = Exception
    sys.modules["openai.error"] = openai.error

from graph_of_thoughts.controller.abstract_language_model import AbstractLanguageModel  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402

from src.methods.prompter import STAGES  # noqa: E402

# Every query() call is constrained to this JSON schema via vLLM's
# structured-outputs decoding, not just prompted for it. Added after two
# independent real parse failures in one validation session: Llama's GoT
# Refine step failed to parse on one run, and Phi-3.5-mini failed on EVERY
# call across all three methods -- not just truncation (completion_tokens
# exactly hit max_tokens), but malformed JSON even in the completed portion
# (it copied multi-line indented evidence blocks, including unescaped
# quotes, verbatim into "command" string values instead of extracting a
# clean single command). A weaker model's tendency to produce syntactically
# invalid JSON is a real, expected difference between models -- but letting
# it silently fail entire runs would confound "the model reasoned badly"
# with "the model can't format JSON," which is a different, less
# interesting failure mode for this project's actual research question.
# Grammar-constrained decoding forces valid JSON syntax (proper escaping,
# required keys, a closed structure) regardless of a model's own formatting
# habits, applied identically to all three models -- it does not fix
# semantic quality (a model can still misidentify or mislabel events), only
# removes syntax as a source of failure.
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "malicious_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "stage": {"type": "string", "enum": STAGES},
                },
                "required": ["command", "stage"],
            },
        },
        "narrative": {"type": "string"},
    },
    "required": ["malicious_events", "narrative"],
}

MODEL_IDS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}

# Phi-4 was the model originally named in idea.md/proposal_document.md, but
# was dropped after validation: its hard 16384-token context ceiling is
# smaller than even WS12 alone (~16.4K tokens under Phi-4's own tokenizer --
# the model failed at the query step, not model load, with "This model's
# maximum context length is 16384 tokens... your prompt contains at least
# 16385 input tokens"), and our largest scenarios run ~58K tokens. Every
# proposal/planning doc was updated to Phi-3.5-mini-instruct (3.8B, 128K
# native context) to match -- see IMPLEMENTATION.md for the full writeup.

# Per-model default gpu_memory_utilization -- cannot be one flat number, and
# checkpoint size is NOT a reliable proxy for how much it needs: smaller
# parameter count does not mean smaller KV cache. Llama needs more than its
# ~15GB weights alone would suggest (at 0.45/~22GB, only 5.1GB was left for
# KV cache, short of the ~8GB its 65536-token max_model_len needs -- raised
# to 0.55). Phi-3.5-mini is the sharpest example: only ~7.6GB of weights,
# but its attention head configuration makes its KV cache far more expensive
# per token than Llama/Qwen's -- 65536 tokens needs ~24GB of KV cache alone,
# so the naive "small model -> small budget" guess of 0.35 (~16.8GB total)
# failed outright ("24.01 GiB KV cache is needed... available: 8.72 GiB").
# Raised to 0.70 to actually cover weights + real KV cache demand.
MODEL_GPU_MEMORY_UTILIZATION = {
    "llama3.1-8b": 0.55,
    "qwen2.5-7b": 0.45,
    "phi3.5-mini": 0.70,
}

# Per-model default max_model_len -- also cannot be one flat number. Forcing
# a larger context than a model actually supports is unsafe (degraded
# quality at best, a hard error at worst -- see the Phi-4 note above), so
# this reflects each model's real, verified ceiling, not a knob to tune away.
#
# Qwen's vLLM-detected default is only 32768 -- fine for WS12 (~16K tokens)
# and Ubuntu (~19K), but APT29/Sidewinder/FIN6 are ~58K tokens (they share
# one W10 log; see IMPLEMENTATION.md Phase 2), which would exceed it.
# Qwen2.5-7B-Instruct's own config supports much longer context via rope
# scaling, so raising it to match Llama's cap is expected to be safe -- but
# this hasn't been empirically verified yet on this project's larger
# scenarios. If it errors, the fix is scenario-specific truncation, not a
# silent lower cap that would make CoT unable to see full evidence.
MODEL_MAX_MODEL_LEN = {
    "llama3.1-8b": 65536,  # native 131072, capped down to fit the KV cache budget
    "qwen2.5-7b": 65536,  # native default 32768, raised -- see note above
    "phi3.5-mini": 65536,  # native 128K -- far more room than needed, capped down only to bound KV cache size
}


class VLLMLanguageModel(AbstractLanguageModel):
    """
    One instance = one loaded model, reused across every LLM call within a
    single reasoning-method run (and can be reused across many runs for the
    same model, to avoid repeated ~20-40s load times). Accumulates
    prompt_tokens / completion_tokens / cost (here: wall-clock seconds,
    since inference is free on our own GPU) across every call, exactly like
    the framework's own Controller.output_graph() expects to read them from
    an AbstractLanguageModel instance.
    """

    def __init__(
        self,
        model_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        seed: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
    ):
        # None means "use this model's known-safe default" (see
        # MODEL_GPU_MEMORY_UTILIZATION / MODEL_MAX_MODEL_LEN above) --
        # explicit args still override per-call if a caller ever needs to.
        if gpu_memory_utilization is None:
            gpu_memory_utilization = MODEL_GPU_MEMORY_UTILIZATION[model_key]
        if max_model_len is None:
            max_model_len = MODEL_MAX_MODEL_LEN[model_key]
        self.logger = None  # AbstractLanguageModel.__init__ sets this properly below
        self.model_key = model_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.num_calls = 0
        self.wall_clock_s = 0.0
        self.parse_retries = 0

        super().__init__(config_path=None, model_name=model_key, cache=False)
        # AbstractLanguageModel.__init__ tries to load a config.json we don't
        # use -- skip that machinery entirely rather than fighting it.
        self.config = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

        model_id = MODEL_IDS[model_key]
        # This is a shared GPU machine with other users' processes already
        # resident (confirmed via nvidia-smi). vLLM's default
        # gpu_memory_utilization=0.9 is a fraction of TOTAL device memory, not
        # free memory, so it would try to grab it regardless of what else is
        # already running there -- a good way to OOM another tenant. Both
        # gpu_memory_utilization and max_model_len are per-model (see
        # MODEL_GPU_MEMORY_UTILIZATION / MODEL_MAX_MODEL_LEN docstrings above)
        # rather than one flat number, because the three models differ enough
        # in checkpoint size (Phi-4's 27.3GB vs. ~14-15GB for the other two)
        # and native context window (Phi-4's 16384 vs. Llama's 131072) that a
        # single value either OOMs one model or wastes headroom on another --
        # both failure modes were hit directly while validating this.
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def load_config(self, path):
        # Overridden no-op: this project passes model config directly via
        # __init__, not the framework's config.json file convention.
        self.config = {}

    def query(self, query: str, num_responses: int = 1, structured: bool = True) -> List[str]:
        # structured=True (the default) is what every framework-driven call
        # gets, since graph_of_thoughts' Operation._execute calls lm.query()
        # generically with no way to pass extra args -- Generate/Aggregate/
        # Improve all want the {malicious_events, narrative} schema. The one
        # exception is ToT's scoring_function (methods/tot.py), which calls
        # lm.query() directly and passes structured=False, since it asks for
        # a single confidence float, not this object shape -- forcing the
        # schema there would silently corrupt the score.
        params = SamplingParams(
            n=num_responses,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            # A textbook fix for a textbook degenerate-decoding failure:
            # Phi-3.5-mini at temperature=0 (greedy) correctly extracted one
            # clean command, then repeated that exact same JSON object
            # verbatim until the token budget ran out, never progressing --
            # a classic LLM repetition trap, distinct from the whitespace-
            # padding failure at temperature=0.7 (see disable_any_whitespace
            # above). Applied to every model, not just Phi-3.5-mini: a mild
            # repetition penalty is standard practice and doesn't advantage
            # any one model's reasoning, only discourages verbatim looping.
            repetition_penalty=1.15,
            structured_outputs=(
                # disable_any_whitespace: a known failure mode of grammar-
                # constrained JSON decoding -- when a weaker model's own next-
                # token preference conflicts with the grammar (here: Phi-3.5-
                # mini kept trying to continue text that the grammar had
                # already forced closed with a premature string-terminating
                # quote), whitespace is often the only token still grammar-
                # valid at every position, so the model can get stuck
                # emitting nothing but whitespace for the rest of the
                # budget -- observed directly (completion_tokens pegged at
                # exactly max_tokens, response tail was 1000+ tab
                # characters). Disabling optional whitespace removes that
                # escape hatch, forcing genuine structural progress instead.
                StructuredOutputsParams(json=_OUTPUT_SCHEMA, disable_any_whitespace=True)
                if structured
                else None
            ),
        )
        start = time.perf_counter()
        [output] = self.llm.chat([{"role": "user", "content": query}], params, use_tqdm=False)
        elapsed = time.perf_counter() - start

        self.num_calls += 1
        self.wall_clock_s += elapsed
        self.prompt_tokens += len(output.prompt_token_ids)
        self.completion_tokens += sum(len(o.token_ids) for o in output.outputs)
        self.cost = self.wall_clock_s  # no dollar cost on local inference -- track time instead

        return [o.text for o in output.outputs]

    def get_response_texts(self, query_responses) -> List[str]:
        return query_responses  # query() already returns List[str]

    def cost_snapshot(self) -> dict:
        return {
            "llm_calls": self.num_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "parse_retries": self.parse_retries,
        }
