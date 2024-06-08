# **讓 Phi-3 成為業界專家**

要將 Phi-3 model 引入產業中，您需要將產業業務數據添加到 Phi-3 model 中。我們有兩個不同的選項，第一個是 RAG (Retrieval Augmented Generation)，第二個是 Fine Tuning。

## **RAG vs Fine-Tuning**

### **檢索增強生成**

RAG 是資料檢索 + 文字生成。企業的結構化資料和非結構化資料存儲在向量資料庫中。當搜尋相關內容時，找到相關摘要和內容以形成上下文，並結合 LLM/SLM 的文字補全能力來生成內容。

### **微調**


微調是基於改進某個模型。它不需要從模型演算法開始，但需要不斷累積數據。如果您想在行業應用中使用更精確的術語和語言表達，微調是您的更好選擇。但如果您的數據經常變動，微調可能會變得複雜。

### **如何選擇**

1. 如果我們的答案需要引入外部資料，RAG 是最佳選擇

2. 如果需要輸出穩定且精確的行業知識，微調將是一個好的選擇。RAG 優先拉取相關內容，但可能無法總是掌握專業的細微差別。

3. 微調需要高品質的資料集，如果只是小範圍的資料，效果不會有太大差異。RAG 更具彈性

4. 微調是一個黑盒子，一種形而上學，很難理解其內部機制。但 RAG 可以更容易找到資料的來源，從而有效地調整幻覺或內容錯誤，並提供更好的透明度。

### **情境**

1. 垂直行業需要特定的專業詞彙和表達，***Fine-tuning*** 將是最佳選擇

2. QA 系統，涉及不同知識點的綜合，***RAG*** 將是最佳選擇

3. 自動化業務流程的組合 ***RAG + Fine-tuning*** 是最佳選擇

## **如何使用 RAG**

![rag](../../imgs/04/01/RAG.png)

A vector database is a collection of data stored in mathematical form. Vector databases make it easier for machine learning models to remember previous inputs, enabling machine learning to be used to support use cases such as search, recommendations, and text generation. Data can be identified based on similarity metrics rather than exact matches, allowing computer models to understand the context of the data.

向量資料庫是一種以數學形式儲存的資料集合。向量資料庫使機器學習模型更容易記住先前的輸入，從而使機器學習能夠支援搜尋、推薦和文本生成等使用情境。資料可以根據相似度指標而非精確匹配來識別，這使得電腦模型能夠理解資料的上下文。

Vector 資料庫是實現 RAG 的關鍵。我們可以通過向量模型（如 text-embedding-3、jina-ai-embedding 等）將資料轉換為向量儲存。

了解更多關於建立 RAG 應用程式 [https://github.com/microsoft/Phi-3CookBook](https://github.com/microsoft/Phi-3CookBook?WT.mc_id=aiml-138114-kinfeylo)

## **如何使用微調**

常用於微調的演算法是 Lora 和 QLora。如何選擇？

- [了解更多，請參閱此範例筆記本](../../code/04.Finetuning/Phi_3_Inference_Finetuning.ipynb)
- [Python 微調範例](../../code/04.Finetuning/FineTrainingScript.py)

### **Lora 和 QLora**

![lora](../../imgs/04/01/qlora.png)

LoRA （Low-Rank Adaptation）和 QLoRA （Quantized Low-Rank Adaptation）都是使用參數高效微調（PEFT）來微調大型語言模型（LLM）的技術。PEFT 技術旨在比傳統方法更高效地訓練模型。
LoRA 是一種獨立的微調技術，通過對權重更新矩陣應用低秩近似來減少記憶體佔用。它提供了快速的訓練時間，並保持了接近傳統微調方法的性能。

QLoRA 是 LoRA 的擴展版本，結合了量化技術以進一步減少記憶體使用量。QLoRA 將預訓練 LLM 中權重參數的精度量化為 4 位元精度，這比 LoRA 更具記憶體效率。然而，由於額外的量化和反量化步驟，QLoRA 訓練比 LoRA 訓練慢約 30%。

QLoRA 使用 LoRA 作為輔助工具來修正量化錯誤期間引入的錯誤。QLoRA 使得在相對較小且高度可用的 GPU 上微調具有數十億參數的龐大模型成為可能。例如，QLoRA 可以使用僅 2 個 GPU 微調需要 36 個 GPU 的 70B 參數模型。

