# **使用 Azure AI Studio 微調 Phi-3**

讓我們探索如何使用 Azure AI Studio 微調 Microsoft 的 Phi-3 Mini 語言模型。微調允許您將 Phi-3 Mini 調整為特定任務，使其變得更強大且具上下文感知能力。

## 考慮事項：

- **功能:** 哪些模型可以進行微調？基礎模型可以被微調成什麼樣子？
- **成本:** 微調的定價模式是什麼？
- **自訂性:** 我可以在多大程度上修改基礎模型 – 以及以何種方式修改？
- **便利性:** 微調實際上是如何進行的 – 我需要撰寫自訂程式碼嗎？我需要自備運算資源嗎？
- **安全性:** 已知微調模型存在安全風險 – 是否有任何防護措施來防止意外傷害？

![AIStudio Models](../../imgs/05/AIStudio/AistudioModels.png)

這裡是開始的步驟：

## 使用 Azure AI Studio 微調 Phi-3

![Finetune AI Studio](../../imgs/05/AIStudio/AIStudiofinetune.png)

**設定您的環境**

Azure AI Studio：如果您還沒有，請登入 [Azure AI Studio](https://ai.azure.com?WT.mc_id=aiml-138114-kinfeylo)。

**建立一個新專案**

點擊 “New” 並建立一個新專案。根據您的使用案例選擇適當的設定。

### 資料準備

**資料集選擇**

收集或建立與您的任務相符的資料集。這可以是聊天指令、問答對或任何相關的文字資料。

**資料預處理**

清理和預處理您的資料。移除雜訊、處理遺漏值，並將文本標記化。

## 模型選擇

**Phi-3 Mini**

你將會微調預訓練的 Phi-3 Mini 模型。確保你能夠存取模型檢查點（例如，"microsoft/Phi-3-mini-4k-instruct"）。

**微調設定**

Hyperparameters: 定義超參數，例如學習率、批次大小和訓練時期數量。

**Loss Function 損失函式**

選擇適合您任務的損失函式（例如，cross-entropy）。

**優化器**

選擇一個最佳化器（例如，Adam）來在訓練期間進行梯度更新。

**微調過程**

- 載入 Pre-Trained Model：載入 Phi-3 Mini checkpoint。
- 添加 Custom Layers：添加任務特定的層（例如，用於聊天指令的分類頭）。

**訓練 模型** 
使用您準備的數據集微調模型。監控訓練進度並根據需要調整超參數。

**評估和驗證**

Validation Set: 將你的資料分成訓練集和驗證集。

**評估效能**

使用像 accuracy、F1-score 或 perplexity 這樣的指標來評估模型表現。

## 儲存微調模型

**Checkpoint** 
儲存微調後的模型檢查點以供未來使用。

## 部署

- 部署為 Web 服務：將您微調過的模型部署為 Azure AI Studio 中的 web 服務。
- 測試端點：向已部署的端點發送測試查詢以驗證其功能。

## 反覆運算與改進

Iterate: 如果效能不令人滿意，請透過調整超參數、增加更多資料或微調額外的訓練週期來進行反覆調整。

## 監控和改進

持續監控模型的行為並根據需要進行改進。

## 自訂和擴展

Custom Tasks: Phi-3 Mini 可以針對聊天指令以外的各種任務進行微調。探索其他使用案例！
Experiment: 嘗試不同的架構、層組合和技術來提升性能。

***注意***: 微調是一個反覆的過程。實驗、學習並調整你的模型，以達到針對你特定任務的最佳結果！

