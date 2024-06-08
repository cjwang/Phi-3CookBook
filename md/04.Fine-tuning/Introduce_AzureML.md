# **介紹 Azure Machine Learning 服務**

[Azure Machine Learning](https://ml.azure.com?WT.mc_id=aiml-138114-kinfeylo) 是一個用於加速和管理機器學習 (ML) 專案生命週期的雲端服務。

ML 專業人士、資料科學家和工程師可以在他們的日常工作流程中使用它來：

- 訓練和部署模型。
管理機器學習操作 (MLOps)。
- 您可以在 Azure Machine Learning 中建立模型，或使用從開源平台（如 PyTorch、TensorFlow 或 scikit-learn）建構的模型。
- MLOps 工具幫助您監控、重新訓練和重新部署模型。

## Azure Machine Learning 是為誰設計的？

**資料科學家和 ML 工程師**

他們可以使用工具來加速和自動化他們的日常工作流程。
Azure ML 提供公平性、可解釋性、追蹤和可審計性的功能。
應用程式開發人員：
他們可以將模型無縫整合到應用程式或服務中。

**平台開發者**

他們可以使用由耐用的 Azure Resource Manager API 支援的一套強大工具。
這些工具允許建構先進的 ML 工具。

**企業**

在 Microsoft Azure 雲端中工作，企業受益於熟悉的安全性和基於角色的存取控制。
設定專案以控制對受保護資料和特定操作的存取。

## 提升團隊每個人的生產力

ML 專案通常需要具備多種技能的團隊來建構和維護。

Azure ML 提供的工具使您能夠：

- 與您的團隊通過共享筆記本、計算資源、無伺服器計算、資料和環境進行協作。
- 開發具有公平性、可解釋性、追蹤性和審計性的模型，以滿足譜系和審計合規要求。
- 快速輕鬆地大規模部署 ML 模型，並通過 MLOps 高效地管理和治理它們。
- 使用內建的治理、安全性和合規性在任何地方執行機器學習工作負載。

## 跨相容性平台工具

任何在 ML 團隊中的人都可以使用他們偏好的工具來完成工作。
無論你是在執行快速實驗、超參數調整、建構管道，還是管理推論，你都可以使用熟悉的介面，包括：

- Azure Machine Learning Studio
- Python SDK（v2）
- Azure CLI（v2）
- Azure Resource Manager REST APIs

隨著您完善模型並在整個開發週期中進行協作，您可以在 Azure Machine Learning studio UI 中分享和查找資產、資源和指標。

## **LLM/SLM 在 Azure ML**

Azure ML 已經新增了許多 LLM/SLM 相關的函式，結合 LLMOps 和 SLMOps 建立一個企業級的生成式人工智慧技術平台。

### **模型目錄**

Enterprise users can deploy different models according to different business scenarios through Model Catalog, and provide services as Model as Service for enterprise developers or users to access.
企業用戶可以通過 Model Catalog 根據不同的業務場景部署不同的模型，並提供 Model as Service 服務供企業開發者或用戶訪問。

![模型](../../imgs/04/03/models.png)

Azure Machine Learning studio 的模型目錄是發現和使用各種模型的中心，使您能夠建構生成式 AI 應用程式。模型目錄包含數百個來自模型提供者的模型，如 Azure OpenAI service、Mistral、Meta、Cohere、Nvidia、Hugging Face，包括由 Microsoft 訓練的模型。來自 Microsoft 以外提供者的模型是非 Microsoft 產品，根據 Microsoft 的產品條款定義，並受模型附帶條款的約束。

### **工作管道**

核心的機器學習管道是將完整的機器學習任務拆分成多步驟的工作流程。每個步驟都是一個可管理的元件，可以單獨開發、優化、配置和自動化。步驟之間通過定義良好的介面連接。Azure Machine Learning 管道服務會自動協調管道步驟之間的所有相依性。

在微調 SLM / LLM 時，我們可以通過 Pipeline 管理我們的資料、訓練和生成過程

![finetuning](../../imgs/04/03/finetuning.png)

### **Prompt flow**

Benefits of using Azure Machine Learning prompt flow
Azure Machine Learning prompt flow offers a range of benefits that help users transition from ideation to experimentation and, ultimately, production-ready LLM-based applications:


**Prompt engineering 敏捷性**


互動式撰寫體驗：Azure Machine Learning prompt flow 提供流程結構的視覺表示，讓使用者能輕鬆理解和導航他們的專案。它還提供類似筆記本的程式碼撰寫體驗，以便高效地進行流程開發和除錯。
提示調整的變體：使用者可以建立和比較多個提示變體，促進迭代的改進過程。

Evaluation: 內建 evaluation 流程使使用者能夠評估其提示和流程的品質和效果。

綜合資源：Azure Machine Learning prompt flow 包含一個內建工具、範例和範本的函式庫，作為開發的起點，激發創意並加速過程。

**企業級 LLM 基於應用程式的準備度**

Collaboration: Azure Machine Learning prompt flow 支援團隊協作，允許多個使用者共同合作進行 prompt 工程專案，共享知識並維護版本控制。

All-in-one 平台：Azure Machine Learning prompt flow 簡化了整個 prompt engineering 流程，從開發和評估到部署和監控。使用者可以輕鬆地將他們的 flow 部署為 Azure Machine Learning endpoint 並實時監控其效能，確保最佳運行和持續改進。

Azure Machine Learning Enterprise Readiness Solutions: Prompt flow leverages Azure Machine Learning 的強大企業就緒解決方案，提供一個安全、可延展且可靠的基礎，用於開發、實驗和部署 flows。

With Azure Machine Learning prompt flow，使用者可以釋放他們的提示工程靈活性，有效地協作，並利用企業級解決方案來成功開發和部署基於 LLM 的應用程式。

結合 Azure ML 的計算能力、資料和不同元件，企業開發人員可以輕鬆建構自己的人工智慧應用程式。

