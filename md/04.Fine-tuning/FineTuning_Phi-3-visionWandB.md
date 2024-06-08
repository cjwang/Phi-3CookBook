# Phi-3-Vision-128K-Instruct 專案概述

## 模型

The Phi-3-Vision-128K-Instruct，一個輕量級、最先進的多模態模型，是此專案的核心。它是 Phi-3 模型家族的一部分，支援長度達到 128,000 個 tokens 的上下文。該模型在包含合成資料和經過仔細篩選的公開網站的多樣化數據集上進行訓練，強調高品質、需要推理的內容。訓練過程包括監督微調和直接偏好優化，以確保精確遵循指令，以及強大的安全措施。

## 建立樣本資料對於幾個原因來說是至關重要的：

1. **測試**: 範例資料允許您在各種情境下測試您的應用程式，而不影響真實資料。這在開發和預備階段尤為重要。

2. **效能調整**: 使用模擬真實資料規模和複雜度的範例資料，您可以識別效能瓶頸並相應地優化您的應用程式。

3. **原型設計**: 範例資料可用於建立原型和模型，有助於理解用戶需求並獲取反饋。

4. **資料分析**: 在資料科學中，範例資料常用於探索性資料分析、模型訓練和算法測試。

5. **安全性**: 在開發和測試環境中使用範例資料可以幫助防止敏感真實資料的意外洩漏。

6. **學習**: 如果您正在學習一項新技術或工具，使用範例資料可以提供一種實際應用所學知識的方法。

記住，範例資料的品質會顯著影響這些活動。它在結構和變異性方面應盡可能接近真實資料。

### Sample Data 建立

[產生 DataSet 腳本](./CreatingSampleData.md)

## 資料集

一個好的範例資料集是 [DBQ/Burberry.Product.prices.United.States dataset](https://huggingface.co/datasets/DBQ/Burberry.Product.prices.United.States)（可在 Huggingface 上找到）。
Burberry 產品的範例資料集以及產品類別、價格和標題的 metadata，共有 3,040 行，每行代表一個獨特的產品。這個資料集讓我們可以測試模型理解和解釋視覺資料的能力，生成捕捉複雜視覺細節和品牌特徵的描述性文字。

**注意：** 您可以使用包含圖像的任意資料集。

## 複雜推理

模型需要僅根據圖像來推理價格和命名。這要求模型不僅要識別視覺特徵，還要理解它們在產品價值和品牌方面的含義。通過從圖像中合成準確的文本描述，該專案突顯了整合視覺資料以提升模型在現實世界應用中的性能和多功能性的潛力。

## Phi-3 Vision Architecture

The model architecture is a multimodal version of a Phi-3. It processes both text and image data, integrating these inputs into a unified sequence for comprehensive understanding and generation tasks. The model uses separate embedding layers for text and images. Text tokens are converted into dense vectors, while images are processed through a CLIP vision model to extract feature embeddings. These image embeddings are then projected to match the text embeddings' dimensions, ensuring they can be seamlessly integrated.

模型架構是 Phi-3 的多模態版本。它處理文本和圖像資料，將這些輸入整合到一個統一的序列中，以進行全面的理解和生成任務。該模型對文本和圖像使用單獨的嵌入層。文本標記被轉換為密集向量，而圖像則通過 CLIP 視覺模型處理以提取特徵嵌入。這些圖像嵌入隨後被投射以匹配文本嵌入的維度，確保它們可以無縫整合。

## Text 和 Image 嵌入的整合

特殊標記在文本序列中指示應插入圖像嵌入的位置。在處理過程中，這些特殊標記會被相應的圖像嵌入取代，使模型能夠將文本和圖像作為單一序列來處理。我們數據集的提示使用特殊的 <|image|> 標記格式如下：

```python
text = f"<|user|>\n<|image_1|>這張圖片顯示的是什麼？<|end|><|assistant|>\n產品: {row['title']}，類別: {row['category3_code']}，全價: {row['full_price']}<|end|>"
```

## 範例程式碼

- [Phi-3-Vision 訓練腳本](../../code/04.Finetuning/Phi-3-vision-Trainingscript.py)
- [Weights and Bias 範例操作說明](https://wandb.ai/byyoung3/mlnews3/reports/How-to-fine-tune-Phi-3-vision-on-a-custom-dataset--Vmlldzo4MTEzMTg3)

