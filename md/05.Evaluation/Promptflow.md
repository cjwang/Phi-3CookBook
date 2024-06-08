# **介紹 Promptflow**

[Microsoft Prompt Flow](https://microsoft.github.io/promptflow/index.html?WT.mc_id=aiml-138114-kinfeylo) 是一個視覺化工作流程自動化工具，允許使用者使用內建範本和自訂連接器來建立自動化工作流程。它旨在讓開發者和業務分析師能夠快速建構自動化流程，用於資料管理、協作和流程優化等任務。使用 Prompt Flow，使用者可以輕鬆連接不同的服務、應用程式和系統，並自動化複雜的業務流程。

Microsoft Prompt Flow 是設計來簡化由大型語言模型 (LLMs) 驅動的 AI 應用程式的端到端開發週期。無論您是在構思、原型設計、測試、評估或部署基於 LLM 的應用程式，Prompt Flow 都能簡化過程，並使您能夠建構具有生產品質的 LLM 應用程式。

## 這裡是使用 Microsoft Prompt Flow 的主要功能和優點：

**互動式創作體驗**

Prompt Flow 提供了您的 flow 結構的視覺表示，使您能夠輕鬆理解和導航您的專案。
它提供了類似筆記本的程式碼編寫體驗，以便高效地進行 flow 開發和除錯。

**Prompt Variants and Tuning**

建立並比較多個提示變體，以促進迭代改進過程。評估不同提示的效能並選擇最有效的提示。

**內建 評估 流程**
使用 內建 評估 工具 評估 您的 提示 和 流程 的 質量 和 效果。
了解 您的 基於 LLM 的 應用程式 的 執行情況。

**綜合資源**

Prompt Flow 包含一個內建工具、範例和模板的函式庫。這些資源作為開發的起點，激發創意，加速過程。

**協作和企業準備**

支持團隊協作，允許多個使用者共同合作進行 prompt 工程專案。
維持版本控制並有效分享知識。簡化整個 prompt 工程流程，從開發和評估到部署和監控。

## 評估在 Prompt Flow

在 Microsoft Prompt Flow 中，評估在評估您的 AI 模型表現方面起著至關重要的作用。讓我們來探討如何在 Prompt Flow 中自訂評估流程和度量標準：

![PFVizualise](../../imgs/05/PromptFlow/pfvisualize.png)

**理解 Prompt Flow 中的評估**

在 Prompt Flow 中，flow 代表處理輸入並產生輸出的節點序列。評估 flow 是一種特殊類型的 flow，旨在根據特定標準和目標評估執行的性能。

**評估流程的主要特點**

他們通常在測試流程之後執行，使用其輸出。他們計算分數或指標來衡量測試流程的性能。指標可以包括準確性、相關性分數或任何其他相關的測量。

### 自訂評估流程

**定義輸入**

Evaluation flows 需要接收正在測試的 執行 輸出。 定義 輸入 與標準流程類似。
例如，如果你正在評估一個 QnA 流程，將一個 輸入 命名為“answer”。如果評估一個分類流程，將一個 輸入 命名為“category”。也可能需要真實標籤 輸入（例如，實際標籤）。

**輸出和指標**

評估流程產生的結果可以衡量被測試流程的效能。指標可以使用 Python 或 LLM (Large Language Models) 來計算。使用 log_metric() 函式來記錄相關的指標。

**使用自訂評估流程**

開發您自己的評估流程，以符合您的特定任務和目標。根據您的評估目標自訂指標。
將這個自訂的評估流程應用於批次執行，以進行大規模測試。

## 內建 評估 方法

Prompt Flow 也提供內建的評估方法。
您可以提交批次執行並使用這些方法來評估您的流程在大數據集上的表現。
查看評估結果，比較指標，並根據需要進行迭代。
請記住，評估對於確保您的 AI 模型符合預期標準和目標至關重要。探索官方文件以獲取有關在 Microsoft Prompt Flow 中開發和使用評估流程的詳細說明。

總結來說，Microsoft Prompt Flow 透過簡化提示工程並提供強大的開發環境，使開發者能夠建立高品質的 LLM 應用程式。如果您正在使用 LLM，Prompt Flow 是一個值得探索的有價值工具。探索 [Prompt Flow 評估文件](https://learn.microsoft.com/azure/machine-learning/prompt-flow/how-to-develop-an-evaluation-flow?view=azureml-api-2?WT.mc_id=aiml-138114-kinfeylo) 以獲取有關在 Microsoft Prompt Flow 中開發和使用評估流程的詳細說明。

