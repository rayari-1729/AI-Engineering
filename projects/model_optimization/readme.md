Wan2.2 8bit : [Link](https://huggingface.co/spaces/r3gm/wan2-2-fp8da-aoti-preview-2/tree/main)


# QNN Execution Paradigms (JIT vs. AoT)
When deploying neural networks to Qualcomm Snapdragon NPUs using the Qualcomm Neural Network (QNN) SDK, performance and deployment flexibility are governed by whether the model graph is evaluated via Just-in-Time (JIT) or Ahead-of-Time (AoT) compilation.

## 1. Just-in-Time (JIT) Execution Path
This paradigm applies when deploying a raw, framework-agnostic model graph directly to the target environment.
* Typical Stack: Raw Model (.onnx) $\rightarrow$ ONNX Runtime (using the QNN Execution Provider).
* Compilation Point: On the target client device at application initialization runtime.
* Mechanism: The high-level runtime parses the ONNX computational graph dynamically, determines the available execution hardware capabilities via the device’s local QNN library, and translates operators into Hexagon NPU machine code on the fly.
* Trade-offs:
    * 🟢 High Portability: A single .onnx binary can target multiple generations of Snapdragon SoCs dynamically.
    * 🔴 Initialization Overhead: Introduces a significant "warm-up" delay (first-frame latency spike) during the initial session setup due to graph compilation overhead.
 
> Note: This overhead is typically mitigated in production via Context Caching, where the engine serializes and dumps the compiled graph to the local disk after the first run, acting as a hybrid AoT path on subsequent boots.

## 2. Ahead-of-Time (AoT) Execution Path
This paradigm applies when compiling the model graph into machine instructions during the development/CI-CD pipeline before application packaging.
* Typical Stack: Raw Model (.onnx) $\rightarrow$ qnn-model-lib-generator $\rightarrow$ Target Toolchain C++ Compiler $\rightarrow$ Hardware-Specific Context Binary (.so / .dll).
* Compilation Point: On the developer workstation or CI/CD runner prior to deployment.
* Mechanism: The model graph is passed through Qualcomm’s offline compilation binaries. The compiler executes layout transformations, memory allocations, operator fusions, and precision optimizations tailored to a specific version of the Hexagon Tensor Processor (HTP).
* Trade-offs:

  * 🟢 Deterministic Performance: Zero runtime optimization or translation overhead. The target device loads the compiled context binary directly into NPU memory instantly.

  * 🔴 Architecture Fragmentation: The resulting binary is strictly tied to specific Hexagon architecture generations (e.g., v73 for Snapdragon 8 Gen 2, v75 for Snapdragon 8 Gen 3, v81/v85 for Snapdragon X Series). Forward compilation targets are rigid and unsupported on legacy hardware.
 
## Quick Reference Matrix

| Metric | JIT Pipeline (High-Level Runtime) | AoT Pipeline (Native QNN Binary) |
| :--- | :--- | :--- |
| **Shipped Asset** | Standard `.onnx` Graph | Pre-compiled Context Binary (`.so` / `.dll`) |
| **Compilation Node** | User Target Device | Developer / CI Workstation |
| **App Initialization** | Delayed (During first-run graph processing) | Instantaneous |
| **Hardware Agnosticism**| Flexible across compatible runtimes | Restricted to target HTP Architecture (e.g., `v73`, `v75`) |
