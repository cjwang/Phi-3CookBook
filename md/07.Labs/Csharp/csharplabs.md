## 歡迎使用 C# 的 Phi-3 實驗室。

有一系列的實驗室展示了如何在 .NET 環境中整合強大的不同版本的 Phi-3 模型。

## 先決條件

在 執行 範例 之前，請確保您已安裝以下內容：

**.NET 8:** 確保您的機器上已安裝[最新版本的 .NET](https://dotnet.microsoft.com/download/dotnet/8.0)。

**（選擇性）Visual Studio 或 Visual Studio Code：** 您將需要一個能夠執行 .NET 專案的 IDE 或程式碼編輯器。建議使用 [Visual Studio](https://visualstudio.microsoft.com/) 或 [Visual Studio Code](https://code.visualstudio.com/)。

**使用 git** 複製本地其中一個可用的 Phi-3 版本從 [Hugging Face](https://huggingface.co).

**下載 phi3-mini-4k-instruct-onnx 模型** 到你的本機:

### 移動到資料夾以儲存模型

```bash
cd c:\phi3\模型
```

### 增加對 lfs 的支援

```bash
git lfs install 
```

### 複製和下載 mini 4K instruct 模型

```bash
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
```

### 複製和下載 vision 128K 模型

```
git clone https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu
```

**重要：** 目前的示範設計為使用 ONNX 版本的模型。之前的步驟會複製以下模型。

![OnnxDownload](../../../imgs/07/00/DownloadOnnx.png)

## 關於實驗室

主要解決方案有幾個範例實驗室，展示了使用 C# 的 Phi-3 模型的功能。

 Project | Description | Location |
| ------------ | ----------- | -------- |
| LabsPhi301    | 這是一個使用本地 phi3 模型來提問的範例專案。該專案使用 `Microsoft.ML.OnnxRuntime` 函式庫載入本地 ONNX Phi-3 模型。 | .\src\LabsPhi301\ |
| LabsPhi302    | 這是一個使用 Semantic Kernel 實作的 Console 聊天範例專案。 | .\src\LabsPhi302\ |
| LabsPhi303 | 這是一個使用本地 phi3 視覺模型來分析影像的範例專案。該專案使用 `Microsoft.ML.OnnxRuntime` 函式庫載入本地 ONNX Phi-3 視覺模型。 | .\src\LabsPhi303\ |
| LabsPhi304 | 這是一個使用本地 phi3 視覺模型來分析影像的範例專案。該專案使用 `Microsoft.ML.OnnxRuntime` 函式庫載入本地 ONNX Phi-3 視覺模型。該專案還提供了一個菜單，具有不同的選項以與使用者互動。 | .\src\LabsPhi304\ 

## 如何執行這些專案

要執行這些專案，請按照以下步驟：

1. 複製這個儲存庫到你的本機。

1. 打開終端機並導航到所需的專案。例如，讓我們執行 `LabsPhi301`。
    ```bash
    cd .\src\LabsPhi301\
    ```

1. 使用以下命令執行專案
    ```bash
    dotnet run
    ```

1. 這個範例專案會要求使用者輸入並使用本地模式回應。

    執行中的示範類似於這個：

    ![Chat running demo](../../../imgs/07/00/SampleConsole.gif)

    ***注意：** 第一個問題有一個錯字，Phi-3 足夠酷炫，會分享正確答案！*

1. 這個專案 `LabsPhi304` 要求使用者選擇不同的選項，然後處理請求。例如，分析本地圖片。

    執行中的示範類似於這個：

    ![Image Analysis running demo](../../../imgs/07/00/SampleVisionConsole.gif)

