# **在 LM Studio 中使用 Phi-3**

[LM Studio](https://lmstudio.ai) 是一個在本地桌面應用程式中呼叫 SLM 和 LLM 的應用程式。它允許使用者輕鬆使用不同的模型，並支援使用 NVIDIA/AMD GPU/Apple Silicon 進行加速運算。通過 LM Studio，使用者可以下載、安裝和執行各種基於 Hugging Face 的開源 LLM 和 SLM，在本地測試模型性能而無需編寫程式碼。

## **1. 安裝**

![LMStudio](../../imgs/02/LMStudio/LMStudio.png)

您可以選擇通過 LM Studio 的網站 [https://lmstudio.ai/](https://lmstudio.ai/) 在 Windows、Linux、macOS 上安裝

## **2. 下載 Phi-3 在 LM Studio**

LM Studio 使用量化的 gguf 格式呼叫開源模型。你可以直接從 LM Studio Search UI 提供的平台下載，或者你可以自行下載並指定在相關目錄中呼叫。

***我們在 LM Studio 搜尋 Phi3 並下載 Phi-3 gguf 模型***

![LMStudioSearch](../../imgs/02/LMStudio/LMStudio_Search.png)

***管理已下載的模型透過 LM Studio***

![LMStudioLocal](../../imgs/02/LMStudio/LMStudio_Local.png)

## **3. 與 Phi-3 在 LM Studio 聊天**

我們在 LM Studio Chat 中選擇 Phi-3 並設定聊天模板（Preset - Phi3）以開始與 Phi-3 的本地聊天

![LMStudio 聊天](../../imgs/02/LMStudio/LMStudio_Chat.png)

***注意***:

a. 您可以通過 LM Studio 控制面板中的進階設定來設置參數

b. 因為 Phi-3 有特定的 Chat 模板要求，必須在 Preset 中選擇 Phi-3

c. 你也可以設定不同的參數，例如 GPU 使用情況等。

## **4. 從 LM Studio 呼叫 Phi-3 API**

LM Studio 支援快速部署本地服務，您可以在不編寫程式碼的情況下建構模型服務。

![LMStudioServer](../../imgs/02/LMStudio/LMStudio_Server.png)

這是 Postman 中的結果

![LMStudioPostman](../../imgs/02/LMStudio/LMStudio_Postman.png)

