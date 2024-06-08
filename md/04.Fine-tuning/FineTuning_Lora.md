# **使用 Lora 微調 Phi-3**

微調 Microsoft 的 Phi-3 Mini 語言模型，使用 [LoRA (Low-Rank Adaptation)](https://github.com/microsoft/LoRA?WT.mc_id=aiml-138114-kinfeylo) 在自訂聊天指令資料集上。

LORA 將有助於改善對話理解和回應生成。

## 如何微調 Phi-3 Mini 的逐步指南:

**Imports 和 設定**

安裝 loralib

```
pip install loralib
# 或者
# pip install git+https://github.com/microsoft/LoRA

```

開始匯入必要的函式庫，例如 datasets、transformers、peft、trl 和 torch。
設定日誌以追蹤訓練過程。

你可以選擇通過用 loralib 中實現的對應部分替換一些層來進行調整。我們目前只支援 nn.Linear、nn.Embedding 和 nn.Conv2d。我們還支援 MergedLinear，用於單個 nn.Linear 代表多個層的情況，例如在一些注意力 qkv 投影的實現中（請參閱附加說明以獲取更多資訊）。

```
# ===== Before =====
# layer = nn.Linear( in_features, out_features)
```

```
# ===== 之後 ======
```

import loralib as lora

```
# 添加一對低階適應矩陣，秩 r=16
layer = lora.Linear(in_features, out_features, r=16)
```

在 training 迴圈開始之前，僅將 LoRA 參數標記為可訓練。

```
import loralib as lora
model = BigModel()
# 這會將所有名稱中不包含字串 "lora_" 的參數的 requires_grad 設為 False
lora.mark_only_lora_as_trainable(model)
# 訓練迴圈
for batch in dataloader:
```

當儲存檢查點時，生成僅包含 LoRA 參數的 state_dict。

```
# ===== 之前 =====
# torch.save(model.state_dict(), checkpoint_path)
```

```
# ===== 之後 =====
torch.save(lora.lora_state_dict(model), checkpoint_path)
```

當使用 load_state_dict 載入檢查點時，請確保設置 strict=False。

```
# 先載入預訓練的檢查點
model.load_state_dict(torch.load('ckpt_pretrained.pt'), strict=False)
# 然後載入 LoRA 檢查點
model.load_state_dict(torch.load('ckpt_lora.pt'), strict=False)
```

現在訓練可以照常進行。

**超參數**

定義兩個字典：training_config 和 peft_config。training_config 包含訓練的超參數，例如學習率、批次大小和日誌設定。

peft_config 指定了 LoRA 相關的參數，如 rank、dropout 和 task type。

**模型和 Tokenizer 載入**

指定預訓練 Phi-3 model 的路徑（例如："microsoft/Phi-3-mini-4k-instruct"）。設定 model 設定，包括快取使用、資料類型（混合精度使用 bfloat16）、以及注意力實作。

**訓練**

微調 Phi-3 模型使用自訂聊天指令資料集。利用來自 peft_config 的 LoRA 設定進行高效適應。使用指定的日誌策略監控訓練進度。
評估和保存：評估微調後的模型。
在訓練期間保存檢查點以供日後使用。

**範例**

- [了解更多這個範例筆記本](../../code/04.Finetuning/Phi_3_Inference_Finetuning.ipynb)
- [Python 微調範例](../../code/04.Finetuning/FineTrainingScript.py)
- [使用 LORA 進行 Hugging Face Hub 微調範例](../../code/04.Finetuning/Phi-3-finetune-lora-python.ipynb)
- [使用 QLORA 進行 Hugging Face Hub 微調範例](../../code/04.Finetuning/Phi-3-finetune-qlora-python.ipynb)

