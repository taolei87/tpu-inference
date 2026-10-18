# glm5_next_ref: GLM-5.3-Flash parity oracle

Test-only reference for the tpu-inference GLM-5.3-Flash (`glm5_next`) model. Nothing under `tpu_inference/` imports it.

| File | What |
|---|---|
| `ref_model.py` | Functional JAX model: KDA, NoPE MLA + DSA k-pool indexer, mHC, MoE, MTP layer. Single device, fp32 by default. |
| `ref_checkpoint.py` | Real-checkpoint naming, the FP8 rule, FP8 block (de)quantization, HF-module-name → real-name conversion, and a reader that loads a checkpoint directory into `ref_model` params. |
| `make_tiny_ckpt.py` | Writes tiny random checkpoints in the real format (`config.json` + `model.safetensors`, optional FP8). |
| `data/real_config.json` | `config.json` of `zai-org/GLM-5.3-Flash` (fetched 2026-10-17). |
| `data/real_weight_templates.json` | Tensor-name templates of the real `model.safetensors.index.json` (76,108 tensors), with which ones are FP8 and on which layers. |

Tests: `tests/models/jax/test_glm5_next_ref.py`. They check `ref_model` against transformers `Glm5NextTextModel`, and the tiny checkpoints against the real naming and FP8 rules.

## Provenance

`ref_model.py` is a trimmed copy of `glm53-flash/engine/glm53/model.py` from the kaggle-tpu-lab repository at commit `9a31b85`. The rest is new code written for these tests. The upstream checkpoint reader (`checkpoint.py`, `hf_convert.py`) informed `ref_checkpoint.py`, but nothing from it was copied.

What the trim removed from upstream `model.py`:

- tensor and sequence sharding (`tp_reduce`, `seq_shard`, collectives);
- the int8 latent cache (`cache_q8`, packing, scale gathers);
- the Pallas kernel switches (`pallas_kda`, `pallas_mhc`);
- speculative-decode histories and rollback (`hist`, `kda_replay`, `rollback_states`);
- per-row batching (`*_rows`);
- the key-dedup fast path (`union_pools`) and pool-window gathers;
- the selection buckets;
- host-resident or grouped expert application.

What else changed:

- `moe_route` picks weights with `take_along_axis` instead of a one-hot sum. The values are identical.
- `hc_pre`'s Sinkhorn uses `sum` instead of the unrolled slice adds. The math is the same; the summation order differs at rounding level.
- `forward` takes `n_layers` for truncated-model tests.
- `mla_project` computes the indexer head weights in f32. vLLM does the same, because bf16 flips near-tie pool rankings at long context.

To re-sync, diff the functions named in `ref_model.py` against upstream `model.py`, then re-run the tests.

### Upstream license (MIT)

```text
MIT License

Copyright (c) 2026 Abdur Rahim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`data/real_config.json` and `data/real_weight_templates.json` are derived from the `zai-org/GLM-5.3-Flash` model repository, which is MIT-licensed.
