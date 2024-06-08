# **在 iOS 中推論 Phi-3**

Phi-3-mini 是 Microsoft 的一個新系列模型，能夠在邊緣設備和物聯網設備上部署大型語言模型（LLM）。Phi-3-mini 可用於 iOS、Android 和邊緣設備的部署，允許生成式 AI 部署在 BYOD。以下範例基於 iOS 部署 Phi-3-mini

## **1. 準備**

a. macOS 14+

b. Xcode 15+

c. iOS SDK 17.x（iPhone 14 A16 或更高）

d. 安裝 Python 3.10+（推薦使用 Conda）

e. 安裝 Python 函式庫 - python-flatbuffers

f. 安裝 CMake

### Semantic Kernel 和 推理：

Semantic Kernel 是一個應用程式框架，允許你建立與 Azure OpenAI Service、OpenAI 模型，甚至本地模型相容的應用程式。透過 Semantic Kernel 存取本地服務，讓你可以輕鬆連接到你自建的 Phi-3-mini 模型伺服器。

### 呼叫量化模型與 Ollama 或 LlamaEdge:

許多使用者偏好使用量化模型來在本地執行模型。[Ollama](https://ollama.com) 和 [LlamaEdge](https://llamaedge.com) 允許個別使用者呼叫不同的量化模型：

**Ollama**
你可以直接執行 ollama run phi3 或離線設定。建立一個 Modelfile，並將 gguf 檔案的路徑寫入其中。以下是執行 Phi-3-mini 量化模型的範例程式碼：

```

FROM {新增你的 gguf 檔案路徑}
TEMPLATE \"\"\"<|user|> {{.Prompt}}<|end|> <|assistant|>\"\"\"
PARAMETER stop <|end|>
PARAMETER num_ctx 4096

```

**LlamaEdge**
如果你想同時在雲端和邊緣設備上使用 gguf，LlamaEdge 可以是你的選擇。

## **2. 編譯 ONNX Runtime for iOS**

```bash

git clone https://github.com/microsoft/onnxruntime.git

cd onnxruntime

./build.sh --build_shared_lib --ios --skip_tests --parallel --build_dir ./build_ios --ios --apple_sysroot iphoneos --osx_arch arm64 --apple_deploy_target 17.4 --cmake_generator Xcode --config Release

```

***注意***

a. 在編譯之前，你必須確保 Xcode 已正確設定，並在終端機上進行設定

```bash

sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer 

```

b. ONNX Runtime 需要基於不同平台編譯。對於 iOS，你可以基於 arm64 / x86_64 進行編譯

c. 建議直接使用最新的 iOS SDK 進行編譯。當然，你也可以降低版本以相容過去的 SDK。

## **3. 使用 ONNX Runtime 編譯生成式 AI 於 iOS**

***注意：*** 因為使用 ONNX Runtime 的生成式 AI 處於預覽階段，請注意變更。

```bash

git clone https://github.com/microsoft/onnxruntime-genai

cd onnxruntime-genai

git checkout yguo/ios-build-genai


mkdir ort

cd ort

mkdir include

mkdir lib

cd ../


cp ../onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h ort/include
cp ../onnxruntime/build_ios/Release/Release-iphoneos/libonnxruntime*.dylib* ort/lib

python3 build.py --parallel --build_dir ./build_ios_simulator --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deployment_target 17.4 --cmake_generator Xcode

```

## **4. 在 Xcode 中建立一個 App 應用程式**

I chose Objective-C as the App development method , because using Generative AI with ONNX Runtime C++ API, Objective-C is better compatible. Of course, you can also complete related calls through Swift bridging.

我選擇 Objective-C 作為 App 開發方法，因為使用 Generative AI 與 ONNX Runtime C++ API，Objective-C 更具相容性。當然，你也可以通過 Swift bridging 完成相關呼叫。

![xcode](../../imgs/03/iOS/xcode.png)

## **5. 複製 ONNX 量化 INT4 模型到 App 應用程式專案**

我們需要匯入 INT4 量化模型（ONNX 格式），需要先下載

![hf](../../imgs/03/iOS/hf.png)

下載後，您需要將其添加到 Xcode 專案的 Resources 目錄中。

![模型](../../imgs/03/iOS/model.png)

## **6. 在 ViewControllers 中新增 C++ API**

***注意***:

a. 將對應的 C++ 標頭檔案新增到專案中

![head](../../imgs/03/iOS/head.png)

b. 在 Xcode 中新增 onnxruntime-gen ai 動態函式庫

![lib](../../imgs/03/iOS/lib.png)

c. 直接使用 C 範例中的程式碼來測試這些範例。你也可以直接新增更多來執行（例如 ChatUI）

d. 因為你需要呼叫 C++，請將 ViewController.m 更改為 ViewController.mm

```objc

    NSString *llmPath = [[NSBundle mainBundle] resourcePath];
    char const *modelPath = llmPath.cString;

    auto model =  OgaModel::Create(modelPath);

    auto tokenizer = OgaTokenizer::Create(*model);

    const char* prompt = "<|system|>You are a helpful AI assistant.<|end|><|user|>Can you introduce yourself?<|end|><|assistant|>";

    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt, *sequences);

    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 100);
    params->SetInputSequences(*sequences);

    auto output_sequences = model->Generate(*params);
    const auto output_sequence_length = output_sequences->SequenceCount(0);
    const auto* output_sequence_data = output_sequences->SequenceData(0);
    auto out_string = tokenizer->Decode(output_sequence_data, output_sequence_length);
    
    auto tmp = out_string;

```

## **7. 執行結果**

![result](../../imgs/03/iOS/result.jpg)

***範例程式碼：*** https://github.com/Azure-Samples/Phi-3MiniSamples/tree/main/ios

