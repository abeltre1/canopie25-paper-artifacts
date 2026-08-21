"""The API serves the CANONICAL model id, not the staged path.

FIELD (Kimi-K3, 8-node MI300A): the serve staged hf://moonshotai/Kimi-K3 onto
the shared FS and handed vLLM the path — /v1/models then advertised
/mnt/models/moonshotai-kimi-k3, and a request naming the id the user actually
typed 404'd. boxy knows both names at plan time; vLLM's --served-model-name is
the knob. The injection lives in build_vllm_serve_cmd (the single choke point:
local, distributed-head and agentless serves all compose the argv there), so
every serve path gets the alias with zero flags.
"""

from boxy import engines
from boxy.box import Box
from boxy.location import Location, Resources, Staging


def _box(**kw) -> Box:
    kw.setdefault("name", "vllm")
    kw.setdefault("engine", "vllm")
    kw.setdefault("image", "vllm/vllm-openai:v0.9.1")
    kw.setdefault("entrypoint", "vllm")
    kw.setdefault("ports", [8000])
    return Box(**kw)


def _loc() -> Location:
    return Location(
        name="t",
        scheduler="slurm",
        accelerator="cuda",
        runtime="podman",
        resources=Resources(nodes=1, gpus_per_node=1),
        staging=Staging(models_dir="./models"),
    )


def test_transport_uri_model_served_under_canonical_id():
    # local/delegated serve: box.model keeps the hf:// spec while the engine is
    # handed the store mount — the id is derived right in the builder.
    cmd = engines.build_vllm_serve_cmd(
        _box(model="hf://meta-llama/Llama-3.1-8B-Instruct"),
        _loc(),
        "/mnt/models/meta-llama-llama-3.1-8b-instruct",
    )
    assert "--served-model-name=meta-llama/Llama-3.1-8B-Instruct" in cmd


def test_rewritten_model_uses_recorded_served_name():
    # agentless prestage rewrites box.model to the staged path and records the
    # original id in served_name — the dash slug alone can't be reversed.
    cmd = engines.build_vllm_serve_cmd(
        _box(model="/scratch/u/boxy/models/moonshotai-kimi-k3",
             served_name="moonshotai/Kimi-K3"),
        _loc(),
        "/mnt/models/moonshotai-kimi-k3",
    )
    assert "--served-model-name=moonshotai/Kimi-K3" in cmd


def test_user_served_model_name_wins():
    # skip-if-present: a post-`--` value is already in the argv when the
    # tack-on runs, so boxy's derived alias never overrides it.
    cmd = engines.build_vllm_serve_cmd(
        _box(model="hf://org/Name"),
        _loc(),
        "/mnt/models/org-name",
        extra_args=["--served-model-name", "my-alias"],
    )
    assert cmd.count("--served-model-name") == 1  # the user's two-token pair
    assert not any(a.startswith("--served-model-name=") for a in cmd)


def test_box_args_served_model_name_wins():
    cmd = engines.build_vllm_serve_cmd(
        _box(model="hf://org/Name", args={"served_model_name": "site-alias"}),
        _loc(),
        "/mnt/models/org-name",
    )
    assert "--served-model-name=site-alias" in cmd
    assert "--served-model-name=org/Name" not in cmd


def test_path_model_without_provenance_is_not_guessed():
    # a bare-path serve has no reliable id (the store slug is ambiguous) —
    # keep the engine's default rather than invent one.
    cmd = engines.build_vllm_serve_cmd(
        _box(model="/data/checkpoints/my-model"), _loc(), "/mnt/models/my-model"
    )
    assert not any("--served-model-name" in a for a in cmd)


def test_engine_pull_bare_id_needs_no_alias():
    # engine-pull mode: `vllm serve <repo id>` already advertises the id itself.
    cmd = engines.build_vllm_serve_cmd(
        _box(model="meta-llama/Llama-3.1-8B-Instruct"),
        _loc(),
        "meta-llama/Llama-3.1-8B-Instruct",
    )
    assert not any("--served-model-name" in a for a in cmd)


def test_llamacpp_is_untouched():
    box = Box(
        name="l",
        engine="llama.cpp",
        image="ghcr.io/ggml-org/llama.cpp:server",
        entrypoint="",
        ports=[8090],
        model="hf://org/name.gguf",
    )
    cmd = engines.build_serve_cmd(box, _loc(), "/mnt/models/name.gguf")
    assert not any("--served-model-name" in str(a) for a in cmd)
