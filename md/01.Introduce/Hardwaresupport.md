## Phi-3 Hardware 支援

Microsoft Phi-3 已經針對 ONNX Runtime 進行了優化，並支援 Windows DirectML。它在各種硬體類型上運行良好，包括 GPU、CPU，甚至是行動裝置。

具體來說，支援的硬體包括：

- GPU SKU: RTX 4090 (DirectML)
- GPU SKU: 1 A100 80GB (CUDA)
- CPU SKU: Standard F64s v2 (64 vCPUs, 128 GiB 記憶體)

**Mobile SKU**

- Android - Samsung Galaxy S21
- Apple iPhone 14 或更高版本 A16/A17 Processor

- 最低配置要求：
- Windows: 支援 DirectX 12 的 GPU 和至少 4GB 的合併 RAM

CUDA：NVIDIA GPU 具有 Compute Capability >= 7.02

![HardwareSupport](../../imgs/00/phi3hardware.png)

隨時在 [Azure AI Studio](https://ai.azure.com) 進一步探索 Phi-3

