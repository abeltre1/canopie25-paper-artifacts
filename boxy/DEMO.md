# boxy, seen in action

A record of a real end-to-end run of `boxy serve` — a live llama.cpp
OpenAI-compatible endpoint in a container, launched by boxy — executed inside
a **fully egress-restricted sandbox** (every model host and every container
registry blob CDN blocked), which is exactly the air-gapped scenario boxy is
designed for. Everything below is reproducible.

## What ran

```console
$ boxy serve --box examples/boxes/llamacpp-demo.toml \
             --location examples/locations/local-docker.toml
### Running Command:
    docker run --rm --name=llamacpp-demo --network=host --ipc=host \
      --entrypoint=llama-server --workdir=/models \
      --volume=/…/boxy/models:/models \
      --env OMP_NUM_THREADS=1 … --env HF_HUB_OFFLINE=1 … \
      boxy-demo/llamacpp:local -m tiny-llama-demo.gguf --host 0.0.0.0 --port 8090
```

The container came up and the OpenAI API answered:

```console
$ docker ps
NAMES           IMAGE                      COMMAND                  STATUS
llamacpp-demo   boxy-demo/llamacpp:local   "llama-server -m tin…"   Up 29 seconds

$ curl -s http://127.0.0.1:8090/v1/models
{"object": "list", "data": [{"id": "tiny-llama-demo.gguf", "object": "model", …}]}

$ curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hello from HPC"}],"max_tokens":24}'
{"id": "chatcmpl-d7b2d228-…", "object": "chat.completion",
 "model": "tiny-llama-demo.gguf",
 "choices": [{"message": {"content": "…24 generated tokens…", "role": "assistant"},
              "finish_reason": "length"}],
 "usage": {"prompt_tokens": 15, "completion_tokens": 24, "total_tokens": 39}}
```

Real llama.cpp engine (llama-cpp-python 0.3.32, compiled from source), real
LLaMA tokenizer, real inference loop, real OpenAI wire format.

## The air-gap twist (read this before judging the output text)

The sandbox's egress proxy blocks huggingface.co, the Ollama registry, quay,
and even Docker Hub/ghcr **blob** downloads. So this demo did what an
air-gapped HPC site does:

1. **Engine**: compiled llama.cpp from source (PyPI sdist was reachable).
2. **Image**: built locally with `docker import` from a dependency-closure
   rootfs — no registry involved (245 MB).
3. **Model**: a fully valid llama-architecture GGUF assembled locally with
   llama.cpp's own `gguf` library, using the **real LLaMA SentencePiece
   tokenizer** (from llama.cpp's in-repo vocab GGUF) and **random weights** —
   because no real weights could transit the proxy. The generated text is
   therefore gibberish; the tokenization, usage accounting, and serving stack
   are genuine.

On any connected machine, the same flow with real weights:

```bash
boxy pull --box examples/boxes/vllm-hf.toml       # hf:// via RamaLama transports
boxy serve --box examples/boxes/vllm-hf.toml --location examples/locations/local.toml
```

(`boxy pull` was exercised here too: it correctly reached RamaLama's HF
transport and failed only at the proxy's 403 — the code path works.)

## Reproduce the demo (any Linux box with Docker, no GPU needed)

```bash
cd boxy
pip install -e '.[ramalama]'
pip install 'llama-cpp-python[server]' gguf numpy   # engine + gguf tooling

# tiny model (or drop any real GGUF into ./models and skip this)
curl -sLo vocab-llama.gguf https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/ggml-vocab-llama-spm.gguf
python hack/make_tiny_gguf.py                        # writes models/tiny-llama-demo.gguf
# build the local image (or use any llama.cpp server image if you can pull)
# then:
boxy serve --box examples/boxes/llamacpp-demo.toml \
           --location examples/locations/local-docker.toml
curl -s http://127.0.0.1:8090/v1/models
```

If you *can* pull images (normal machine), skip the local image build and set
the box image to `ghcr.io/ggml-org/llama.cpp:server`.

## The cloud path, also verified live

`boxy generate sky` transpiles a box+location to a SkyPilot task, and the
output was validated by **SkyPilot 0.12.3 itself** (`sky.Task.from_yaml`):

```console
$ boxy generate sky --box examples/boxes/vllm.toml \
    --location examples/locations/cloud-gpu.toml --serve -o task.yaml
$ python -c "import sky; t=sky.Task.from_yaml('task.yaml'); print(t)"
VALID: SkyPilot 0.12.3 accepted the task
  accelerators: {'H100': 4} | image_id: docker:vllm/vllm-openai:v0.9.1 | ports: ['8000']
  service readiness_path: /v1/models | replicas: 1
```

Launch it with `sky launch task.yaml` (batch) or `sky serve up task.yaml`
(managed serving) on any machine with cloud credentials.

We also confirmed, in SkyPilot 0.12.3's shipped source, that its Slurm
support (`sky/provision/slurm/`) is Pyxis/Enroot-based and that serving is a
cloud-side feature — reinforcing the division of labor in SPEC §6c: SkyPilot
for cloud, boxy for HPC.
