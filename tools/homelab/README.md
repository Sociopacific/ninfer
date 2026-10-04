# Single RTX 5090 native session cache profile

This experimental setup uses Qwen3.8-27B NVFP4full, vision, NVFP4 KV, YaRN3 and MTP7. It has a 712704-token technical capacity, prefill chunk1024, one active request, eight private continuations and 32GiB Host KV. The native persistence implementation was exercised with 514988 input tokens plus two images. The full 710K persistence roundtrip remains untested; the GPU memory margin is small.

Build the serve target using `cmake --build build --target ninfer-serve -j` after configuring the repository's CUDA toolchain. Supply your own compatible `.ninfer` artifact; weights and session snapshots are not included.

```sh
python3 tools/homelab/run_native_cache.py --binary /path/to/ninfer-serve --model /path/to/qwen3_8_27b_nvfp4full.ninfer --cache-directory /path/to/session-cache
```

The launcher restores the last published snapshot before HTTP listening when CURRENT exists. Initial startup uses an empty catalog otherwise. This launcher is a portable equivalent of the qualified container argv; the wrapper itself has not been GPU-benchmarked separately.

```sh
curl -sS http://127.0.0.1:18080/cache/save -H 'content-type: application/json' -d '{"directory":"/path/to/session-cache","async":true}'
curl -sS http://127.0.0.1:18080/cache/status
```

Wait for durable status before relying on a snapshot across restart. HTTP202 only acknowledges capture. Generation can proceed during background disk writing. New messages are not automatically saved; old immutable generations have no automatic GC. Directory paths must be accessible to the serving process. Manual `/cache/load` requires an idle, empty continuation catalog. See docs/serving.md for API details.

Observed on the qualified host: initial native D2H10.24GB in0.725s; capture10.40GB in1.47s when main KV was already in Host RAM; background durable save47.64s. Cold disk restore10.57s, warm filesystem-cache restore0.80s; restored515K prompt preparation about0.776s including H2D and five new tokens. These are different phases and workloads, not a uniform latency guarantee. Three independent40/40/48K vision sessions survived restart; three independent500K sessions were not tested. Twenty consecutive truncated imports were rejected, followed by a successful valid import in the same process.

The NVFP4 envelope extension is experimental and marked TEMP-REMOVE in source. Physical cold disk latency, temporary RAM duplication during save, power-loss fault injection and same-length payload corruption remain limitations or untested scenarios.
