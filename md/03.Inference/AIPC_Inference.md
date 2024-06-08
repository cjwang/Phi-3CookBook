# **在 AI PC 中推論 Phi-3**

隨著生成式 AI 的進步和邊緣設備硬體能力的提升，越來越多的生成式 AI 模型現在可以整合到用戶的自帶設備（BYOD）中。AI PC 是這些模型之一。從 2024 年開始，Intel、AMD 和 Qualcomm 已經與 PC 製造商合作，引入 AI PC，通過硬體修改來促進本地生成式 AI 模型的部署。在這次討論中，我們將重點關注 Intel AI PC，並探討如何在 Intel AI PC 上部署 Phi-3。

### **什麼是 NPU**

An NPU (Neural Processing Unit) 是一個專用的處理器或處理單元，位於較大的 SoC 上，專門用於加速神經網路操作和 AI 任務。與通用的 CPU 和 GPU 不同，NPU 針對資料驅動的平行計算進行了優化，使其在處理大量多媒體資料（如影片和影像）以及神經網路資料處理方面非常高效。它們特別擅長處理 AI 相關任務，例如語音識別、視訊通話中的背景模糊，以及物件檢測等照片或影片編輯過程。

## **NPU vs GPU**

雖然許多 AI 和機器學習工作負載在 GPU 上執行，但 GPU 和 NPU 之間有一個關鍵的區別。
GPU 以其平行計算能力而聞名，但並非所有 GPU 在處理圖形之外都同樣高效。另一方面，NPU 是專為神經網路運算中的複雜計算而設計的，使其在 AI 任務中非常有效。

總結來說，NPU 是加速 AI 計算的數學高手，它們在新興的 AI PC 時代中扮演著關鍵角色！

***這個範例基於 Intel 最新的 Intel Core Ultra Processor***

## **1. 使用 NPU 執行 Phi-3 模型**

Intel® NPU 裝置是一種與 Intel 用戶端 CPU 整合的 AI 推理加速器，從 Intel® Core™ Ultra 代 CPU（前稱為 Meteor Lake）開始。它能夠實現人工神經網路任務的高效能執行。

![延遲](../../imgs/03/AIPC/aipcphitokenlatency.png)

![延遲770](../../imgs/03/AIPC/aipcphitokenlatency770.png)

**Intel NPU 加速函式庫**

The Intel NPU Acceleration Library [https://github.com/intel/intel-npu-acceleration-library](https://github.com/intel/intel-npu-acceleration-library) 是一個 Python 函式庫，旨在利用 Intel 神經處理單元 (NPU) 的強大功能，在相容硬體上執行高速計算，以提升您的應用程式效率。

Example of Phi-3-mini 在由 Intel® Core™ Ultra 處理器驅動的 AI PC 上。

![DemoPhiIntelAIPC](../../imgs/03/AIPC/aipcphi3-mini.gif)

安裝 Python 函式庫 使用 pip

```bash

   pip install intel-npu-acceleration-library

```

***注意*** 專案仍在開發中，但參考模型已經非常完整。

### **執行 Phi-3 與 Intel NPU 加速函式庫**

使用 Intel NPU 加速，這個函式庫不會影響傳統的編碼過程。你只需要使用這個函式庫來量化原始的 Phi-3 模型，例如 FP16、INT4，例如

```python

from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM, pipeline
import intel_npu_acceleration_library
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", use_cache=True, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("編譯 NPU 的模型")
model = intel_npu_acceleration_library.compile(model, dtype=torch.float16)

```

在量化成功後，繼續執行以呼叫 NPU 來執行 Phi-3 模型。

```python

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

query = "<|system|>You are a helpful AI assistant.<|end|><|user|>Can you introduce yourself?<|end|><|assistant|>"

output = pipe(query, **generation_args)

output[0]['generated_text']

```

當執行程式碼時，我們可以通過任務管理器查看 NPU 的執行狀態

![NPU](../../imgs/03/AIPC/aipc_NPU.png)

***範例*** : [AIPC_NPU_DEMO.ipynb](../../code/03.Inference/AIPC/AIPC_NPU_DEMO.ipynb)

## **2. 使用 DirectML + ONNX Runtime 執行 Phi-3 Model**

### **什麼是 DirectML**

[DirectML](https://github.com/microsoft/DirectML) 是一個高效能、硬體加速的 DirectX 12 函式庫，用於機器學習。DirectML 提供 GPU 加速，適用於各種常見的機器學習任務，並支援包括 AMD、Intel、NVIDIA 和 Qualcomm 等廠商的所有 DirectX 12 相容 GPU。

當單獨使用時，DirectML API 是一個低階的 DirectX 12 函式庫，適用於高效能、低延遲的應用程式，例如框架、遊戲和其他即時應用程式。DirectML 與 Direct3D 12 的無縫互操作性，以及其低開銷和跨硬體的一致性，使得 DirectML 成為加速機器學習的理想選擇，當需要高效能且結果在跨硬體上具有可靠性和可預測性時尤其如此。

***注意*** : 最新的 DirectML 已經支援 NPU(https://devblogs.microsoft.com/directx/introducing-neural-processor-unit-npu-support-in-directml-developer-preview/)

### DirectML 和 CUDA 在其功能和性能方面：

**DirectML** 是由 Microsoft 開發的機器學習函式庫。它旨在加速 Windows 裝置上的機器學習工作負載，包括桌上型電腦、筆記型電腦和邊緣裝置。

- DX12-Based：DirectML 建構在 DirectX 12（DX12）之上，提供跨 GPU 的廣泛硬體支援，包括 NVIDIA 和 AMD。
- Wider Support：由於它利用了 DX12，DirectML 可以在任何支援 DX12 的 GPU 上運行，甚至是整合型 GPU。
- Image Processing：DirectML 使用神經網路處理圖像和其他資料，適用於圖像識別、物件檢測等任務。
- Ease of Setup：設定 DirectML 非常簡單，不需要 GPU 製造商的特定 SDK 或函式庫。
- Performance：在某些情況下，DirectML 表現良好，甚至在某些工作負載上比 CUDA 更快。
- Limitations：然而，在某些情況下，DirectML 可能會較慢，特別是對於 float16 大批量大小的情況。

**CUDA** 是 NVIDIA 的平行運算平台和程式設計模型。它允許開發人員利用 NVIDIA GPU 的強大功能進行通用計算，包括機器學習和科學模擬。

- NVIDIA-Specific: CUDA 是與 NVIDIA GPU 緊密整合並專為其設計的。
- Highly Optimized: 它為 GPU 加速任務提供了卓越的效能，特別是在使用 NVIDIA GPU 時。
- Widely Used: 許多機器學習框架和函式庫（如 TensorFlow 和 PyTorch）都支援 CUDA。
- Customization: 開發人員可以為特定任務微調 CUDA 設定，這可以帶來最佳效能。
- Limitations: 然而，CUDA 對 NVIDIA 硬體的相依性可能會在您希望跨不同 GPU 獲得更廣泛相容性時帶來限制。

### 選擇 DirectML 和 CUDA 之間：

選擇 DirectML 和 CUDA 取決於您的具體使用案例、硬體可用性和偏好。如果您尋求更廣泛的相容性和簡便的設定，DirectML 可能是一個不錯的選擇。然而，如果您擁有 NVIDIA GPU 並需要高度優化的效能，CUDA 仍然是一個強有力的競爭者。總之，DirectML 和 CUDA 各有其優勢和劣勢，因此在做決定時請考慮您的需求和可用的硬體。

### **生成式 AI 與 ONNX Runtime**

在 AI 時代，AI 模型的可攜性非常重要。ONNX Runtime 可以輕鬆地將訓練好的模型部署到不同的設備上。開發者不需要關注推理框架，使用統一的 API 完成模型推理。在生成式 AI 時代，ONNX Runtime 也進行了程式碼優化（https://onnxruntime.ai/docs/genai/）。通過優化的 ONNX Runtime，可以在不同的終端上推理量化的生成式 AI 模型。在使用 ONNX Runtime 的生成式 AI 中，你可以通過 Python、C#、C / C++ 來推理 AI 模型 API。當然，在 iPhone 上部署可以利用 C++ 的生成式 AI 與 ONNX Runtime API。

[範例 程式碼](https://github.com/Azure-Samples/Phi-3MiniSamples/tree/main/onnx)

***編譯生成式 AI 與 ONNX Runtime 函式庫***

```bash

winget install --id=Kitware.CMake  -e

git clone https://github.com/microsoft/onnxruntime.git

cd .\onnxruntime\

./build.bat --build_shared_lib --skip_tests --parallel --use_dml --config Release

cd ../

git clone https://github.com/microsoft/onnxruntime-genai.git

cd .\onnxruntime-genai\

mkdir ort

cd ort

mkdir include

mkdir lib

copy ..\onnxruntime\include\onnxruntime\core\providers\dml\dml_provider_factory.h ort\include

copy ..\onnxruntime\include\onnxruntime\core\session\onnxruntime_c_api.h ort\include

copy ..\onnxruntime\build\Windows\Release\Release\*.dll ort\lib

copy ..\onnxruntime\build\Windows\Release\Release\onnxruntime.lib ort\lib

python build.py --use_dml

```

**安裝 函式庫**

```bash

pip install .\onnxruntime_genai_directml-0.3.0.dev0-cp310-cp310-win_amd64.whl

```

這是 執行 結果

![DML](../../imgs/03/AIPC/aipc_DML.png)

***範例*** : [AIPC_DirectML_DEMO.ipynb](../../code/03.Inference/AIPC/AIPC_DirectML_DEMO.ipynb)

## **3. 使用 Intel OpenVino 執行 Phi-3 模型**

### **什麼是 OpenVINO**

[OpenVINO](https://github.com/openvinotoolkit/openvino) 是一個開源工具包，用於優化和部署深度學習模型。它為來自 TensorFlow、PyTorch 等流行框架的視覺、音頻和語言模型提供增強的深度學習性能。開始使用 OpenVINO。OpenVINO 也可以與 CPU 和 GPU 結合使用來執行 Phi-3 模型。

***注意***: 目前，OpenVINO 暫時不支援 NPU。

### **安裝 OpenVINO 函式庫**

```bash

 pip install git+https://github.com/huggingface/optimum-intel.git

 pip install git+https://github.com/openvinotoolkit/nncf.git

 pip install openvino-nightly

```

### **執行 Phi-3 與 OpenVINO**

像 NPU 一樣，OpenVINO 通過執行量化模型來完成生成式 AI 模型的呼叫。我們需要先對 Phi-3 模型進行量化，並通過 optimum-cli 在命令列上完成模型量化。

**INT4**

```bash

optimum-cli export openvino --model "microsoft/Phi-3-mini-4k-instruct" --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.6  --sym  --trust-remote-code ./openvinomodel/phi3/int4

```

**FP16**

```bash

optimum-cli export openvino --model "microsoft/Phi-3-mini-4k-instruct" --task text-generation-with-past --weight-format fp16 --trust-remote-code ./openvinomodel/phi3/fp16

```

轉換後的格式，如下所示

![openvino_convert](../../imgs/03/AIPC/aipc_OpenVINO_convert.png)

載入模型路徑（model_dir）、相關配置（ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}）和硬體加速設備（GPU.0）通過 OVModelForCausalLM

```python

ov_model = OVModelForCausalLM.from_pretrained(
     model_dir,
     device='GPU.0',
     ov_config=ov_config,
     config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
     trust_remote_code=True,
)

```

當執行 程式碼 時，我們可以通過 任務管理器 查看 GPU 的 執行 狀態

![openvino_gpu](../../imgs/03/AIPC/aipc_OpenVINO_GPU.png)

***範例*** : [AIPC_OpenVino_Demo.ipynb](../../code/03.Inference/AIPC/AIPC_OpenVino_Demo.ipynb)

### ***注意*** : 上述三種方法各有其優勢，但建議使用 NPU 加速進行 AI PC 推論。

