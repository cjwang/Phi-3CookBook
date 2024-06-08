## 提到的關鍵技術包括：

1. [DirectML](https://learn.microsoft.com/windows/ai/directml/dml?WT.mc_id=aiml-138114-kinfeylo) - 一個基於 DirectX 12 建構的硬體加速機器學習的低階 API。
2. [CUDA](https://blogs.nvidia.com/blog/what-is-cuda-2/) - 一個由 Nvidia 開發的平行運算平台和應用程式介面 (API) 模型，使圖形處理單元 (GPU) 能夠進行通用處理。
3. [ONNX](https://onnx.ai/) (Open Neural Network Exchange) - 一個開放格式，旨在表示機器學習模型，提供不同 ML 框架之間的互操作性。
4. [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) (Generic Graph Update Format) - 一種用於表示和更新機器學習模型的格式，特別適用於能夠在 CPU 上有效執行的 4-8bit 量化的小型語言模型。

### DirectML

DirectML 是一個低階 API，可實現硬體加速的機器學習。它建立在 DirectX 12 之上以利用 GPU 加速，並且與供應商無關，這意味著它不需要更改程式碼即可在不同 GPU 供應商之間運作。它主要用於 GPU 上的模型訓練和推理工作負載。

如同硬體支援，DirectML 被設計用來與廣泛的 GPU 一起工作，包括 AMD 整合和獨立 GPU、Intel 整合 GPU 和 NVIDIA 獨立 GPU。它是 Windows AI 平台的一部分，並且在 Windows 10 和 11 上受到支援，允許在任意 Windows 裝置上進行模型訓練和推論。

有關 DirectML 的更新和機會，例如支援多達 150 個 ONNX 運算子，並且由 ONNX runtime 和 WinML 使用。它由主要的整合硬體供應商（IHVs）支持，每個供應商都實現了各種元命令。

### CUDA

CUDA，即 Compute Unified Device Architecture，是由 Nvidia 創建的平行運算平台和應用程式介面（API）模型。它允許軟體開發人員使用支援 CUDA 的圖形處理單元（GPU）進行通用目的處理——這種方法稱為 GPGPU（General-Purpose computing on Graphics Processing Units）。CUDA 是 Nvidia GPU 加速的關鍵推動力，廣泛應用於各個領域，包括機器學習、科學計算和視頻處理。

Nvidia 的 GPU 專門支援 CUDA 硬體，因為這是 Nvidia 開發的專有技術。每個架構支援特定版本的 CUDA 工具包，該工具包提供開發者建構和執行 CUDA 應用程式所需的函式庫和工具。

### ONNX

ONNX (Open Neural Network Exchange) 是一種開放格式，旨在表示機器學習模型。它提供了一個可擴展的計算圖模型的定義，以及內建運算子和標準資料類型的定義。ONNX 允許開發者在不同的 ML 框架之間移動模型，實現互操作性，並使建立和部署 AI 應用程式變得更加容易。

Phi3 mini 可以在多種裝置上使用 ONNX Runtime 執行，包括伺服器平台、Windows、Linux 和 Mac 桌面，以及行動 CPU。 我們新增的最佳化設定是

- ONNX models for int4 DML: 量化為 int4 透過 AWQ
- ONNX model for fp16 CUDA
- ONNX model for int4 CUDA: 量化為 int4 透過 RTN
- ONNX model for int4 CPU and Mobile: 量化為 int4 透過 RTN

### Llama.cpp

Llama.cpp 是一個用 C++ 編寫的開源軟體函式庫。它對各種大型語言模型（LLMs）進行推理，包括 Llama。與 ggml 函式庫（通用張量函式庫）一起開發，llama.cpp 旨在提供比原始 Python 實現更快的推理速度和更低的記憶體使用量。它支援硬體優化、量化，並提供簡單的 API 和範例。如果你對高效的 LLM 推理感興趣，llama.cpp 值得探索，因為 Phi3 可以執行 llama.cpp。

### GGUF

GGUF（Generic Graph Update Format）是一種用於表示和更新機器學習模型的格式。它對於能夠在具有 4-8bit 量化的 CPU 上有效執行的小型語言模型（SLM）特別有用。GGUF 對於快速原型設計以及在邊緣設備或 CI/CD 管道等批次作業中執行模型非常有益。

