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

# Per-model default gpu_memory_utilization -- cannot be one flat number.
# Model checkpoint sizes differ enough (Llama/Qwen ~14-15GB, Phi-3.5-mini a
# much smaller ~7.6GB) that one flat value either wastes headroom or doesn't
# fit. Llama also needs more here than its weights alone would suggest: at
# 0.45 (~22GB) it had only 5.1GB left for KV cache, not enough for its
# 65536-token max_model_len (needs ~8GB) -- hit this directly before raising
# it to 0.55.
MODEL_GPU_MEMORY_UTILIZATION = {
    "llama3.1-8b": 0.55,
    "qwen2.5-7b": 0.45,
    "phi3.5-mini": 0.35,
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

    def query(self, query: str, num_responses: int = 1) -> List[str]:
        params = SamplingParams(
            n=num_responses,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
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
