# 02. Gemma 4 runs locally with tool calls

_2026-10-23T22:30_

Gemma 4 E2B and E4B on the Mac via Ollama and Google's LiteRT-LM. Given a camera image and six robot tools, every call returned a well-formed tool call. LiteRT-LM also took a spoken command directly as audio: 'turn left' became turn(90).

| Metric | Value |
|---|---|
| E2B decision | 1.05 s |
| E2B decode | 79 tok/s |
| LiteRT E2B first token | 0.42 s |
| well-formed tool calls | 11/11 |

**Say on stage:** The robot's brain is Gemma 4, running entirely on this laptop, no cloud.

- [bench_e2b.txt](bench_e2b.txt)
- [bench_e4b.txt](bench_e4b.txt)
- [litert_bench.txt](litert_bench.txt)
