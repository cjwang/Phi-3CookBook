# **在本地伺服器上推論 Phi-3**

我們可以在本地伺服器上部署 Phi-3。使用者可以選擇 [Ollama](https://ollama.com) 或 [LM Studio](https://llamaedge.com) 解決方案，或者他們可以撰寫自己的程式碼。你可以透過 [Semantic Kernel](https://github.com/microsoft/semantic-kernel?WT.mc_id=aiml-138114-kinfeylo) 或 [Langchain](https://www.langchain.com/) 來連接 Phi-3 的本地服務以建構 Copilot 應用程式

## **使用 Semantic Kernel 存取 Phi-3-mini**

在 Copilot 應用程式中，我們透過 Semantic Kernel / LangChain 建立應用程式。這種類型的應用程式框架通常與 Azure OpenAI Service / OpenAI models 相容，並且也能支援 Hugging Face 上的開源模型和本地模型。如果我們想使用 Semantic Kernel 存取 Phi-3-mini 該怎麼辦？以 .NET 為例，我們可以將其與 Semantic Kernel 中的 Hugging Face Connector 結合使用。預設情況下，它可以對應 Hugging Face 上的 model id（第一次使用時，模型會從 Hugging Face 下載，這需要很長時間）。你也可以連接到已建構的本地服務。相比之下，我們建議使用後者，因為它具有更高的自主性，特別是在企業應用中。

![sk](../../imgs/03/LocalServer/sk.png)

從圖中可以看出，通過 Semantic Kernel 訪問本地服務可以輕鬆連接到自建的 Phi-3-mini 模型伺服器。以下是執行結果

![skrun](../../imgs/03/LocalServer/skrun.png)

***範例程式碼*** https://github.com/kinfey/Phi3MiniSamples/tree/main/semantickernel

