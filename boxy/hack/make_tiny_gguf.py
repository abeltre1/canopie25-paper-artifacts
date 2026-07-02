"""Assemble a tiny, valid llama-architecture GGUF using llama.cpp's own gguf-py.

Tokenizer: copied from llama.cpp's in-repo ggml-vocab-llama-spm.gguf (the real
LLaMA SentencePiece vocab). Weights: random (every public model host is blocked
by this sandbox's egress policy) — the file is a fully valid model that
llama.cpp loads and serves; on a connected system, swap in real weights via
`boxy pull`.
"""

import os

import numpy as np
from gguf import GGUFReader, GGUFWriter

VOCAB_SRC = "vocab-llama.gguf"
OUT = "models/tiny-llama-demo.gguf"

N_VOCAB = 32000
DIM = 64
N_LAYERS = 2
N_HEADS = 8
N_KV_HEADS = 8
FFN = 172  # ~8/3 * dim, rounded to multiple of 4

rng = np.random.default_rng(12345)


def t(shape):
    return (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(np.float32)


os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
w = GGUFWriter(OUT, "llama")
w.add_name("tiny-llama-demo")
w.add_context_length(512)
w.add_embedding_length(DIM)
w.add_block_count(N_LAYERS)
w.add_feed_forward_length(FFN)
w.add_head_count(N_HEADS)
w.add_head_count_kv(N_KV_HEADS)
w.add_rope_dimension_count(DIM // N_HEADS)
w.add_layer_norm_rms_eps(1e-5)
w.add_file_type(0)  # F32

# Copy the real tokenizer metadata wholesale from the vocab-only GGUF.
r = GGUFReader(VOCAB_SRC)
for field in r.fields.values():
    if not field.name.startswith("tokenizer."):
        continue
    vtype = field.types[0]
    sub_type = field.types[1] if len(field.types) > 1 else None
    w.add_key_value(field.name, field.contents(), vtype, sub_type=sub_type)

w.add_tensor("token_embd.weight", t((N_VOCAB, DIM)))
for i in range(N_LAYERS):
    p = f"blk.{i}."
    w.add_tensor(p + "attn_norm.weight", np.ones(DIM, dtype=np.float32))
    w.add_tensor(p + "attn_q.weight", t((DIM, DIM)))
    w.add_tensor(p + "attn_k.weight", t((DIM, DIM)))
    w.add_tensor(p + "attn_v.weight", t((DIM, DIM)))
    w.add_tensor(p + "attn_output.weight", t((DIM, DIM)))
    w.add_tensor(p + "ffn_norm.weight", np.ones(DIM, dtype=np.float32))
    w.add_tensor(p + "ffn_gate.weight", t((FFN, DIM)))
    w.add_tensor(p + "ffn_down.weight", t((DIM, FFN)))
    w.add_tensor(p + "ffn_up.weight", t((FFN, DIM)))
w.add_tensor("output_norm.weight", np.ones(DIM, dtype=np.float32))
w.add_tensor("output.weight", t((N_VOCAB, DIM)))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", OUT)
