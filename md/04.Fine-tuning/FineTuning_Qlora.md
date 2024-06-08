**微調 Phi-3 與 QLoRA**

微調 Microsoft 的 Phi-3 Mini 語言模型，使用 [QLoRA (Quantum Low-Rank Adaptation)](https://github.com/artidoro/qlora)。

QLoRA 將有助於提升對話理解和回應產生。

要使用 transformers 和 bitsandbytes 以 4bits 載入模型，你必須從原始碼安裝 accelerate 和 transformers，並確保你擁有最新版本的 bitsandbytes 函式庫。

**範例**

- [了解更多此範例筆記本](../../code/04.Finetuning/Phi_3_Inference_Finetuning.ipynb)
- [Python 微調範例](../../code/04.Finetuning/FineTrainingScript.py)
- [Hugging Face Hub 使用 LORA 進行微調範例](../../code/04.Finetuning/Phi-3-finetune-lora-python.ipynb)
- [Hugging Face Hub 使用 QLORA 進行微調範例](../../code/04.Finetuning/Phi-3-finetune-qlora-python.ipynb)

