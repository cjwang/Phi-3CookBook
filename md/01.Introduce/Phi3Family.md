﻿# **Phi-3 Family**

Phi-3 模型是目前最強大且具成本效益的小型語言模型（SLM），在各種語言、推理、程式碼和數學基準測試中，表現優於同尺寸和更大尺寸的模型。此版本擴展了高品質模型的選擇，為客戶提供了更多實用的選擇，用於撰寫和建構生成式 AI 應用程式。

Phi-3 模型是目前最強大且具成本效益的小型語言模型 (SLMs)，在各種語言、推理、程式碼和數學基準測試中，表現優於同尺寸和更大尺寸的模型。此次發佈擴展了高品質模型的選擇，為客戶提供了更多實用的選擇，用於撰寫和建構生成式 AI 應用程式。

The Phi-3 Family 包括 mini、小型、中型和 vision 版本，根據不同的參數量進行訓練，以服務各種應用場景。每個模型都經過指令調整，並根據 Microsoft 的負責任 AI、安全和安全標準進行開發，以確保其可以即時使用。

## Phi-3 任務範例

| | |
|-|-|
|任務|Phi-3|
|語言任務|是|
|數學與推理|是|
|程式設計|是|
|函式呼叫|否|
|自我編排（Assistant）|否|
|專用嵌入模型|否

## **Phi-3-Mini**

Phi-3-mini 是一個 3.8B 參數的語言模型，提供兩種上下文長度 [128K](https://aka.ms/phi3-mini-128k-azure-ai) 和 [4K.](https://aka.ms/phi3-mini-4k-azure-ai)

Phi-3-Mini 是一個基於 Transformer 的語言模型，擁有 38 億個參數。它使用包含教育性有用資訊的高品質數據進行訓練，並增強了包含各種 NLP 合成文本的新數據來源，以及內部和外部的聊天數據集，顯著提高了聊天能力。此外，Phi-3-Mini 在預訓練後通過監督微調（SFT）和直接偏好優化（DPO）進行了聊天微調。經過這些後期訓練，Phi-3-Mini 在多項能力上顯示出顯著的改進，特別是在對齊性、穩健性和安全性方面。該模型是 Phi-3 系列的一部分，並以 Mini 版本推出，具有兩個變體，4K 和 128K，代表它可以支持的上下文長度（以 tokens 計）。

## **Phi-3-Small**

Phi-3-small 是一個 70 億參數的語言模型，可用於兩種上下文長度 [128K](https://aka.ms/phi3-small-128k-azure-ai) 和 [8K.](https://aka.ms/phi3-small-8k-azure-ai)

Phi-3-Small 是一個基於 Transformer 的語言模型，擁有 70 億個參數。它使用包含教育性有用資訊的高品質數據進行訓練，並增強了由各種 NLP 合成文本以及內部和外部聊天數據集組成的新數據源，這顯著提升了聊天能力。此外，Phi-3-Small 在預訓練後通過監督微調（SFT）和直接偏好優化（DPO）進行了聊天微調。經過這些後期訓練，Phi-3-Small 在多項能力上顯示出顯著的改進，特別是在對齊性、魯棒性和安全性方面。與 Phi-3-Mini 相比，Phi-3-Small 也在多語言數據集上進行了更密集的訓練。該模型系列提供兩個變體，8K 和 128K，代表它可以支持的上下文長度（以 tokens 計）。

## **Phi-3-Medium**

Phi-3-medium 是一個 14B 參數語言模型，有兩種上下文長度 [128K](https://aka.ms/phi3-medium-128k-azure-ai) 和 [4K.](https://aka.ms/phi3-medium-4k-azure-ai)

Phi-3-Medium 是一個基於 Transformer 的語言模型，擁有 140 億個參數。它使用包含教育性有用資訊的高品質資料進行訓練，並增強了包含各種 NLP 合成文本的新資料來源，以及內部和外部的聊天數據集，顯著提高了聊天能力。此外，Phi-3-Medium 在預訓練後通過監督微調 (SFT) 和直接偏好優化 (DPO) 進行了聊天微調。經過這些後期訓練，Phi-3-Medium 在多項能力上表現出顯著提升，特別是在對齊性、穩健性和安全性方面。該模型系列提供了兩個變體，4K 和 128K，代表它可以支持的上下文長度（以 tokens 計）。

## **Phi-3-vision**

The [Phi-3-vision](https://aka.ms/phi3-vision-128k-azure-ai) 是一個具有語言和視覺能力的 4.2B 參數多模態模型。

Phi-3-vision 是 Phi-3 系列中的第一個多模態模型，將文字和圖像結合在一起。Phi-3-vision 可以用於對現實世界圖像進行推理，並從圖像中提取和推理文字。它還針對圖表和原理圖理解進行了優化，可用於產生見解和回答問題。Phi-3-vision 建立在 Phi-3-mini 的語言能力基礎上，繼續在小尺寸中保持強大的語言和圖像推理質量。

## **Phi Silica**

我們正在介紹 Phi Silica，它是從 Phi 系列模型建構而成，專為 Copilot+ PC 中的 NPU 設計。Windows 是第一個擁有為 NPU 和內建收件箱量身打造的最先進小型語言模型 (SLM) 的平台。
Phi Silica API 以及 OCR、Studio Effects、Live Captions、Recall User Activity API 將在六月的 Windows Copilot 函式庫中提供。更多的 API，例如 Vector Embedding、RAG API、Text Summarization 將在稍後推出。

## **尋找所有 Phi-3 模型**

- [Azure AI](https://aka.ms/phi3-azure-ai)
- [Hugging Face.](https://aka.ms/phi3-hf)

## Model Selection 的範例

| | | | |
|-|-|-|-|
|客戶需求|任務|開始使用|更多細節|
|需要一個簡單總結訊息線索的模型|對話摘要|Phi-3 text model|決定因素是客戶有明確且簡單的語言任務|
|免費的兒童數學輔導應用程式|數學和推理|Phi-3 text models|因為應用程式是免費的，客戶希望解決方案不會產生經常性費用|
|自動巡邏車攝像頭|視覺分析|Phi-Vision|需要一個可以在無網際網路的情況下工作的解決方案|
|想要建構一個基於 AI 的旅行預訂代理|需要複雜的規劃、函式呼叫和協調|GPT models|需要能夠規劃、呼叫 API 以收集資訊並執行|
|想要為員工建構一個輔助工具|RAG、多領域、複雜且開放式|GPT models|開放式場景，需要更廣泛的世界知識，因此更適合使用較大的模型|

