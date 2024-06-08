## 歡迎使用 AI 工具包 for VS Code

[AI Toolkit for VS Code](https://github.com/microsoft/vscode-ai-toolkit/tree/main) 將來自 Azure AI Studio Catalog 和其他如 Hugging Face 的目錄中的各種模型匯集在一起。該工具包通過以下方式簡化了使用生成式 AI 工具和模型建構 AI 應用程式的常見開發任務：

- 開始使用模型探索和 playground。
- 使用本地計算資源進行模型微調和推論。

[安裝 AI Toolkit for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio)

![AIToolkit 精調](../../imgs/04/00/Aitoolkit.png)

**[私人預覽]** 一鍵佈建 Azure Container Apps 以在雲端執行模型微調和推論。

現在讓我們進入你的 AI 應用程式開發：

- [本地開發](#local-development)
    - [準備工作](#preparations)
    - [啟動 Conda](#activate-conda)
    - [僅微調基礎模型](#base-model-fine-tuning-only)
    - [模型微調和推理](#model-fine-tuning-and-inferencing)
- [**[私密預覽]** 遠端開發](#private-preview-remote-development)
    - [先決條件](#prerequisites)
    - [設定遠端開發專案](#setting-up-a-remote-development-project)
    - [配置 Azure 資源](#provision-azure-resources)
    - [[可選] 添加 Huggingface Token 到 Azure Container App Secret](#optional-add-huggingface-token-to-the-azure-container-app-secret)
    - [執行微調](#run-fine-tuning)
    - [配置推理端點](#provision-inference-endpoint)
    - [部署推理端點](#deploy-the-inference-endpoint)
    - [進階用法](#advanced-usage)

## 本地開發

### 準備工作

1. 確保 NVIDIA 驅動程式已安裝在主機上。
2. 如果您使用 HF 來利用資料集，請執行 `huggingface-cli login`。
3. `Olive` 主要設定說明，用於修改記憶體使用量的任何操作。

### 啟用 Conda

由於我們使用 WSL 環境並且是共享的，你需要手動啟動 conda 環境。在這一步之後，你可以執行微調或推理。

```bash
conda activate [conda-env-name] 
```

### 基礎模型微調僅限

要嘗試基礎模型而不進行微調，可以在啟動 conda 後執行此命令。

```bash
cd inference

# Web 瀏覽器介面允許調整一些參數，如最大新 token 長度、溫度等。
# 使用者必須在 gradio 建立連線後，手動在瀏覽器中打開連結（例如：http://0.0.0.0:7860）。
python gradio_chat.py --baseonly
```

### 模型微調和推理

一旦工作區在開發容器中打開，打開終端機（預設路徑是專案根目錄），然後執行下面的命令來微調選定數據集上的 LLM。

```bash
python finetuning/invoke_olive.py 
```

檢查點和最終模型將會儲存在 `models` 資料夾中。

接下來透過 `console`、`web browser` 或 `prompt flow` 中的聊天來執行微調模型的推理。

```bash
cd inference

# 控制台介面。
python console_chat.py

# 網頁瀏覽器介面允許調整一些參數，如最大新標記長度、溫度等。
# 使用者必須在 gradio 建立連接後手動在瀏覽器中打開連結（例如：http://127.0.0.1:7860）。
python gradio_chat.py
```

要在 VS Code 中使用 `prompt flow`，請參考這個[快速開始](https://microsoft.github.io/promptflow/how-to-guides/quick-start.html)。

## **[Private Preview]** 遠端開發

### 先決條件

1. 要在遠端 Azure Container App 環境中執行模型微調，請確保您的訂閱有足夠的 GPU 容量。提交 [support ticket](https://azure.microsoft.com/support/create-ticket/) 以請求應用程式所需的容量。[獲取有關 GPU 容量的更多資訊](https://learn.microsoft.com/en-us/azure/container-apps/workload-profiles-overview)
2. 如果您在 HuggingFace 上使用私人資料集，請確保您有一個 [HuggingFace account](https://huggingface.co/) 並 [generate an access token](https://huggingface.co/docs/hub/security-tokens)
3. 在 AI Toolkit for VS Code 中啟用 Remote Fine-tuning and Inference 功能標誌
   1. 選擇 *File -> Preferences -> Settings* 來打開 VS Code 設定。
   2. 導航到 *Extensions* 並選擇 *AI Toolkit*。
   3. 選擇 *"Enable Remote Fine-tuning And Inference"* 選項。
   4. 重新載入 VS Code 以生效。

### 設定 遠端開發專案

1. 執行指令面板 `AI Toolkit: Focus on Resource View`。
2. 導覽到 *Model Fine-tuning* 以存取模型目錄。為您的專案指定一個名稱並選擇其在您的機器上的位置。然後，點擊 *"Configure Project"* 按鈕。
3. 專案設定
    1. 避免啟用 *"Fine-tune locally"* 選項。
    2. Olive 設定將顯示預設值。請根據需要調整並填寫這些設定。
    3. 移動到 *Generate Project*。這個階段利用 WSL 並涉及設定新的 Conda 環境，為包括 Dev Containers 的未來更新做準備。
4. 點擊 *"Relaunch Window In Workspace"* 以開啟您的遠端開發專案。

> **注意：** 該專案目前可以在本地或遠端的 AI Toolkit for VS Code 中運行。如果在專案建立過程中選擇 *「本地微調」*，它將僅在 WSL 中運行，無法進行遠端開發。另一方面，如果不啟用 *「本地微調」*，該專案將被限制在遠端的 Azure Container App 環境中。

### 設定 Azure 資源

要開始，您需要為遠端微調配置 Azure Resource。請從命令面板執行 `AI Toolkit: Provision Azure Container Apps job for fine-tuning`。

監控提供進度通過輸出通道中顯示的連結。

### [選擇性] 將 Huggingface Token 添加到 Azure Container App Secret

如果你正在使用私有 HuggingFace 資料集，將你的 HuggingFace token 設定為環境變數，以避免在 Hugging Face Hub 上需要手動登入。
你可以使用 `AI Toolkit: Add Azure Container Apps Job secret for fine-tuning command` 來做到這一點。使用此命令，你可以將秘密名稱設置為 [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken) 並使用你的 Hugging Face token 作為秘密值。

### 執行微調

要開始遠端微調工作，執行 `AI Toolkit: Run fine-tuning` 指令。

要查看系統和控制台日誌，您可以使用輸出面板中的連結訪問 Azure 入口網站（更多步驟請參閱 [在 Azure 上查看和查詢日誌](https://aka.ms/ai-toolkit/remote-provision#view-and-query-logs-on-azure)）。或者，您可以通過執行命令 `AI Toolkit: Show the running fine-tuning job streaming logs` 直接在 VSCode 輸出面板中查看控制台日誌。

> **注意：** 由於資源不足，工作可能會被排隊。如果日誌未顯示，執行 `AI Toolkit: Show the running fine-tuning job streaming logs` 指令，等待一會兒，然後再次執行該指令以重新連接到串流日誌。

在此過程中，QLoRA 將用於微調，並將建立 LoRA 適配器以供模型在推理期間使用。
微調的結果將存儲在 Azure Files 中。

### 設定推理端點

在遠端環境中訓練適配器之後，使用簡單的 Gradio 應用程式與模型互動。
類似於微調過程，您需要透過從命令面板執行 `AI Toolkit: Provision Azure Container Apps for inference` 來設定 Azure 資源以進行遠端推論。

預設情況下，用於推理的訂閱和資源群組應與用於微調的訂閱和資源群組匹配。推理將使用相同的 Azure Container App Environment 並訪問儲存在 Azure Files 中的模型和模型適配器，這些都是在微調步驟中生成的。

### 部署推論端點

如果你希望修改推論程式碼或重新載入推論模型，請執行 `AI Toolkit: Deploy for inference` 命令。這將同步你最新的程式碼到 Azure Container App 並重新啟動副本。

一旦部署成功完成，您可以透過點擊 VSCode 通知中顯示的「*Go to Inference Endpoint*」按鈕來訪問推論 API。或者，可以在 `./infra/inference.config.json` 中的 `ACA_APP_ENDPOINT` 和輸出面板中找到 web API 端點。您現在可以使用此端點來評估模型。

### 進階用法

如需有關使用 AI Toolkit 進行遠端開發的更多資訊，請參閱[遠端微調模型](https://aka.ms/ai-toolkit/remote-provision)和[使用微調模型進行推論](https://aka.ms/ai-toolkit/remote-inference)的文件。

