# **在 Ollama 中使用 Phi-3**

[Ollama](https://ollama.com) 允許更多人通過簡單的腳本直接部署開源 LLM 或 SLM，還可以建構 API 來幫助本地 Copilot 應用場景。

## **1. 安裝**

Ollama 支援在 Windows、macOS 和 Linux 上執行。你可以通過這個連結安裝 Ollama（[https://ollama.com/download](https://ollama.com/download)）。成功安裝後，你可以直接使用 Ollama 腳本通過終端視窗呼叫 Phi-3。你可以看到 Ollama 中所有的[可用函式庫。](https://ollama.com/library)

```bash

ollama 執行 phi3

```

***注意:*** 當你第一次執行它時，模型將會先被下載。當然，你也可以直接指定已下載的 Phi-3 模型。我們以 WSL 為例來執行命令。模型成功下載後，你可以直接在終端機上進行互動。

![執行](../../imgs/02/Ollama/ollama_run.png)

## **2. 從 Ollama 呼叫 phi-3 API**

如果你想呼叫由 ollama 產生的 Phi-3 API，你可以在終端機中使用此命令來啟動 Ollama 伺服器。

```bash

ollama serve

```

***注意：*** 如果執行 MacOS 或 Linux，請注意您可能會遇到以下錯誤 <b>"Error: listen tcp 127.0.0.1:11434: bind: address already in use"</b> 您可能在呼叫執行命令時遇到此錯誤。解決此問題的方法是：

**macOS**

```bash

brew services restart ollama

```

**Linux**

```bash

sudo systemctl stop ollama

```

Ollama 支援兩個 API：generate 和 chat。您可以根據需求呼叫 Ollama 提供的模型 API。本地服務埠 11434。例如

**聊天**

```bash

curl http://127.0.0.1:11434/api/chat -d '{
  "model": "phi3",
  "messages": [
    {
      "role": "system",
      "content": "你是一名 Python 開發者。"
    },
    {
      "role": "user",
      "content": "幫我生成一個冒泡排序演算法"
    }
  ],
  "stream": false
  
}'

```

這是在 Postman 中的結果

![聊天](../../imgs/02/Ollama/ollama_chat.png)

```bash

curl http://127.0.0.1:11434/api/generate -d '{
  "model": "phi3",
  "prompt": "<|system|>你是我的 AI 助手。<|end|><|user|>告訴我如何學習 AI<|end|><|assistant|>",
  "stream": false
}'

```

這是在 Postman 中的結果

![gen](../../imgs/02/Ollama/ollama_gen.png)

# 其他資源

查看 Ollama 中可用模型的列表在[此連結。](https://ollama.com/library)

從 Ollama 伺服器拉取您的模型，使用此命令

```bash
ollama pull phi3
```

執行該模型使用此指令

```bash
ollama 執行 phi3
```

***注意：*** 訪問此連結 [https://github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md) 以了解更多資訊

## 呼叫 Ollama 從 JavaScript

```javascript
# Summarize a file with Phi-3 的範例
script({
    model: "ollama:phi3",
    title: "使用 Phi-3 進行摘要",
    system: ["system"],
})

# 摘要範例
const file = def("FILE", env.files)
$`將 ${file} 摘要成一個段落。`
```

## 從 C# 呼叫 Ollama

建立一個新的 C# Console 應用程式並新增以下的 NuGet 套件：

```bash
dotnet add package Microsoft.SemanticKernel --version 1.13.0
```

然後將此程式碼替換到 `Program.cs` 檔案中

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

// 使用本地 ollama 伺服器端點新增聊天完成服務
#pragma warning disable SKEXP0001, SKEXP0003, SKEXP0010, SKEXP0011, SKEXP0050, SKEXP0052
builder.AddOpenAIChatCompletion(
    modelId: "phi3",
    endpoint: new Uri("http://localhost:11434/"),
    apiKey: "non required");

// 呼叫一個簡單的提示給聊天服務
string prompt = "寫一個關於小貓的笑話";
var response = await kernel.InvokePromptAsync(prompt);
Console.WriteLine(response.GetValue<string>());
```

執行應用程式的指令：

```bash
dotnet run
```

