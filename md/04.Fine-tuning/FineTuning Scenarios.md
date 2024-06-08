## 微調場景

| | | | | | | |
|-|-|-|-|-|-|-|
|情境|LoRA|QLoRA|PEFT|DeepSpeed|ZeRO|DORA|
|將預訓練的 LLMs 調整為特定任務或領域|Yes|Yes|Yes|Yes|Yes|Yes|
|針對 NLP 任務（如文本分類、命名實體識別和機器翻譯）進行微調|Yes|Yes|Yes|Yes|Yes|Yes|
|針對 QA 任務進行微調|Yes|Yes|Yes|Yes|Yes|Yes|
|針對聊天機器人生成類人回應進行微調|Yes|Yes|Yes|Yes|Yes|Yes|
|針對生成音樂、藝術或其他形式的創作進行微調|Yes|Yes|Yes|Yes|Yes|Yes|
|降低計算和財務成本|Yes|Yes|No|Yes|Yes|No|
|減少記憶體使用量|No|Yes|No|Yes|Yes|Yes|
|使用較少的參數進行高效微調|No|Yes|Yes|No|No|Yes|
|記憶體高效的數據並行形式，可訪問所有可用 GPU 設備的聚合 GPU 記憶體|No|No|No|Yes|Yes|Yes

## Fine Tuning Performance Examples

![微調效能](../../imgs/04/00/Finetuningexamples.png)

