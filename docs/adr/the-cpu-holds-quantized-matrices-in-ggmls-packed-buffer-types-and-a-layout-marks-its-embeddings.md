# The CPU holds quantized matrices in ggml's packed buffer types, and a layout marks its embeddings

## Context

ggml's CPU backend has buffer types besides its own, one per kind of kernel that wants the weights in an order of its
own: the AMX one on an x86 CPU with AMX, which packs Q8_0 and the K-quants into its tiles' order, and its repacking
of Q4_0 and the like for AVX and NEON. A weight runs through those kernels only when it lives in such a buffer, which
is where llama.cpp puts its weights; a weight in the default buffer runs through ggml's ordinary kernels, with
llamafile's sgemm where it applies. On 8 cores of a Xeon under KVM (2026-10-11), the AMX buffer multiplied a Q8_0
matrix of Irodori-TTS's DiT, 1280 by 3680, by 91 columns in 0.30 ms where the default buffer took 1.26 ms, with the
same values, and so cut the MeanFlow sampler from 0.39 s to 0.23 s and the first audio by 0.2 s. An F16 matrix through
the AMX buffer took longer than through the default one, 1.27 ms against 1.02 ms.

A packed buffer holds the bytes in an order that only its matrix product reads: ggml_get_rows() on a packed embedding
gives garbage, and Qwen3-ASR's token embedding is read by both, as its output matrix. ggml's AMX product also reads
the activation's rows at the weight's width apart, whatever the activation's strides or type say.

## Decision

- **A quantized matrix that ggml_mul_mat() alone reads goes into the first extra buffer type of the device that
  computes a product by it**, asked tensor by tensor through a probe of the tensor's shape and type in a buffer of that
  type, as llama.cpp asks. F16 and F32 tensors stay in the device's default buffer type, since the packed types gain
  nothing on them. A GPU has no extra buffer type, so nothing changes there.
- **The layout tables say which matrices ggml_get_rows() reads.** A matrix takes `quantized_storage()`, an embedding
  `embedding_storage()`, stored in the same types and held in the default buffer type. The family that builds the
  graphs is the one that knows, and its layout already lists every tensor.
- **The matrix-product helper of the library makes the activation contiguous float32** before a product by a weight
  in an extra buffer type, so that no graph of any family has to know which kernel reads it.

The alternatives were turned down:

- Every weight in the extra buffer type, as a whole context. The embeddings would be packed, and the F16 files, which
  Hugging Face ships, would run slower.
- Telling the embeddings apart by name, such as `embd` or `embed`. A name is not a fact about how a tensor is read,
  and a family could name one otherwise.
- Asking the device with the operation each tensor is actually read by. The loader builds no graph and the families
  build theirs after loading, so it would have to be told the same thing the storage says.

## Consequences

The CPU's speed with a quantized file depends on the CPU: on one with AMX, Q8_0 weights make Irodori-TTS's first audio
shorter than F16 weights, and on one without, the two speak as fast. A family that adds a tensor read by
ggml_get_rows() declares it with `embedding_storage()`, or it reads garbage on an x86 CPU with AMX with a quantized
file. `speech quantize` writes the same types for both storages.
