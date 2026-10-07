# Glossary

Short definitions of terms used across the guide. Chapter references point to where each is explained in depth.

| Term | Meaning | Where |
|---|---|---|
| **A2A (Agent2Agent)** | Open protocol for agents to discover each other (Agent Cards) and delegate tasks across frameworks/orgs; v1.0 in 2026 under the Linux Foundation | Ch 6 |
| **ACP** | Ambiguous: IBM's *Agent Communication Protocol* (merged into A2A, Aug 2025) or Zed's *Agent Client Protocol* (editor ↔ coding agent) | Ch 6 |
| **Active parameters** | Parameters actually used per token in an MoE model (vs. total parameters stored) | Ch 4 |
| **APC** | Automatic Prefix Caching in vLLM — reuses KV blocks for identical prefixes | Ch 1 |
| **AWQ / GPTQ** | Post-training 4-bit weight quantization methods | Ch 4 |
| **Compaction** | Rewriting conversation history into a smaller form when context nears its limit | Ch 2 |
| **Context bloat** | Accumulation of low-value tokens in the context window | Ch 2 |
| **Context rot** | Degradation of model accuracy as context length grows | Ch 2 |
| **Continuous batching** | Requests join/leave the running batch at every decode step | Ch 4 |
| **Decode** | Generating output tokens one at a time; memory-bandwidth-bound | Ch 1 |
| **DPI** | Dots per inch when rasterizing a PDF page; determines pixels and therefore visual tokens | Ch 3 |
| **Egress allowlist** | Network policy permitting outbound traffic only to approved domains | Ch 5, 9 |
| **Expert parallelism (EP)** | Sharding MoE experts across GPUs | Ch 4 |
| **GGUF** | llama.cpp model file format with k-quant types (Q8_0, Q4_K_M, …) | Ch 4 |
| **GQA** | Grouped-query attention — fewer KV heads than query heads, shrinking the KV cache | Ch 4 |
| **Indirect prompt injection** | Malicious instructions planted in content an agent reads (web, files, tool outputs) | Ch 9 |
| **KV cache** | Stored Key/Value tensors for previous tokens, reused during decode | Ch 1 |
| **Lethal trifecta** | Private data + untrusted content + external communication in one agent context | Ch 9 |
| **LoRA** | Low-Rank Adaptation — trains a low-rank update BA on frozen weights | Ch 8 |
| **MCP** | Model Context Protocol — JSON-RPC protocol connecting agents to tools, resources, prompts | Ch 6 |
| **MoE** | Mixture of Experts — router activates a few expert FFNs per token | Ch 4 |
| **MRTR** | Multi Round-Trip Requests — MCP 2026-07-28 pattern replacing server→client requests | Ch 6 |
| **Namespaces / cgroups / seccomp** | Linux kernel primitives for isolation, resource limits, and syscall filtering | Ch 5 |
| **NF4** | 4-bit NormalFloat data type used by QLoRA | Ch 8 |
| **PagedAttention** | vLLM's block-based KV memory management | Ch 4 |
| **Prefill** | Processing all prompt tokens in parallel; compute-bound | Ch 1 |
| **Prefix caching** | Reusing KV cache for an identical token prefix across requests | Ch 1 |
| **Progressive disclosure** | Loading information (tool schemas, skill bodies) only when needed | Ch 2, 6 |
| **Pseudonymization** | Replacing identifiers with consistent, reversible tokens | Ch 9 |
| **QLoRA** | LoRA on a 4-bit-quantized frozen base model | Ch 8 |
| **RadixAttention** | SGLang's radix-tree-based KV cache sharing | Ch 1 |
| **Repo map** | Compressed, ranked summary of a codebase's files and symbols | Ch 5 |
| **Skill (Agent Skill)** | Folder with SKILL.md + optional scripts/references, loaded on demand | Ch 6 |
| **SSE** | Server-Sent Events — one-way HTTP streaming (`text/event-stream`) | Ch 7 |
| **Tensor / pipeline / data parallelism** | Splitting layers / splitting layer groups / replicating the model across GPUs | Ch 4 |
| **TTFT** | Time to first token | Ch 1, 4 |
| **Visual tokens** | Tokens a VLM produces from image patches (e.g., one per 28×28 px region) | Ch 3 |
