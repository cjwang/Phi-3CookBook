## 如何使用 Azure ML 系統註冊表中的聊天完成元件來微調模型

在這個範例中，我們將對 Phi-3-mini-4k-instruct 模型進行微調，以使用 ultrachat_200k 資料集完成兩個人之間的對話。

![MLFineTune](../../imgs/04/03/MLFineTune.png)

範例將向您展示如何使用 Azure ML SDK 和 Python 進行微調，然後將微調後的模型部署到線上端點以進行實時推理。

### 訓練資料

我們將使用 ultrachat_200k 資料集。這是 UltraChat 資料集的一個經過嚴格過濾的版本，並用於訓練 Zephyr-7B-β，一個最先進的 7b 聊天模型。

### 模型

我們將使用 Phi-3-mini-4k-instruct 模型來展示使用者如何微調模型以完成聊天任務。如果您是從特定模型卡片打開此筆記本，請記得替換特定模型名稱。

### Tasks 
### 任務

- 選擇一個模型進行微調。
- 選擇並探索訓練資料。
- 配置微調工作。
- 執行微調工作。
- 審查訓練和評估指標。
- 註冊微調後的模型。
- 部署微調後的模型以進行實時推論。
- 清理資源。

## 1. 設定前置條件

- 安裝相依套件
- 連接到 AzureML Workspace。了解更多資訊，請參閱設定 SDK 認證。替換 <WORKSPACE_NAME>、<RESOURCE_GROUP> 和 <SUBSCRIPTION_ID>。
- 連接到 azureml 系統註冊表
- 設定一個可選的實驗名稱
- 檢查或建立運算資源。

Requirements a single GPU node can have multiple GPU cards. For example, in one node of Standard_NC24rs_v3 there are 4 NVIDIA V100 GPUs while in Standard_NC12s_v3, there are 2 NVIDIA V100 GPUs. Refer to the docs for this information. The number of GPU cards per node is set in the param gpus_per_node below. Setting this value correctly will ensure utilization of all GPUs in the node. The recommended GPU compute SKUs can be found here and here.

### Python 函式庫

安裝相依套件，方法是執行以下單元。如果在新環境中執行，這不是可選步驟。

```
pip install azure-ai-ml
pip install azure-identity
pip install datasets==2.9.0
pip install mlflow
pip install azureml-mlflow
```

### 與 Azure ML 互動

這個 Python 程式碼用於與 Azure Machine Learning (Azure ML) 服務互動。以下是它的功能分解：

它從 azure.ai.ml、azure.identity 和 azure.ai.ml.entities 套件匯入必要的模組。它還匯入了 time 模組。

它嘗試使用 DefaultAzureCredential() 進行驗證，這提供了一個簡化的驗證體驗，以快速開始開發在 Azure 雲端中執行的應用程式。如果這失敗，它會退回到 InteractiveBrowserCredential()，這提供了一個互動式的登入提示。

它接著嘗試使用 from_config 方法建立一個 MLClient 實例，該方法從預設的 config 檔案（config.json）中讀取配置。如果這失敗了，它會通過手動提供 subscription_id、resource_group_name 和 workspace_name 來建立一個 MLClient 實例。

它會建立另一個 MLClient 實例，這次是為名為 "azureml" 的 Azure ML 註冊表。這個註冊表是用來儲存模型、微調管道和環境的地方。

它將 experiment_name 設定為 "chat_completion_Phi-3-mini-4k-instruct"。

它通過將當前時間（自紀元以來的秒數，作為浮點數）轉換為整數，然後轉換為字串來生成唯一的時間戳。此時間戳可用於建立唯一的名稱和版本。

```
# Import necessary modules from Azure ML and Azure Identity
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)
from azure.ai.ml.entities import AmlCompute
import time  # Import time module

# Try to authenticate using DefaultAzureCredential
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:  # If DefaultAzureCredential fails, use InteractiveBrowserCredential
    credential = InteractiveBrowserCredential()

# Try to create an MLClient instance using the default config file
try:
    workspace_ml_client = MLClient.from_config(credential=credential)
except:  # If that fails, create an MLClient instance by manually providing the details
    workspace_ml_client = MLClient(
        credential,
        subscription_id="<SUBSCRIPTION_ID>",
        resource_group_name="<RESOURCE_GROUP>",
        workspace_name="<WORKSPACE_NAME>",
    )

# Create another MLClient instance for the Azure ML registry named "azureml"
# This registry is where models, fine-tuning pipelines, and environments are stored
registry_ml_client = MLClient(credential, registry_name="azureml")

# Set the experiment name
experiment_name = "chat_completion_Phi-3-mini-4k-instruct"

# Generate a unique timestamp that can be used for names and versions that need to be unique
timestamp = str(int(time.time()))
```

## 2. 選擇一個基礎模型進行微調

Phi-3-mini-4k-instruct 是一個 3.8B 參數、輕量級、最先進的開放模型，基於用於 Phi-2 的數據集構建。該模型屬於 Phi-3 模型家族，Mini 版本有兩個變體 4K 和 128K，這是它可以支持的上下文長度（以 tokens 計），我們需要為我們的特定目的微調該模型才能使用。您可以在 AzureML Studio 的 Model Catalog 中瀏覽這些模型，通過 chat-completion 任務進行篩選。在此示例中，我們使用 Phi-3-mini-4k-instruct 模型。如果您為不同的模型打開了此 notebook，請相應地更改模型名稱和版本。

注意模型 id 屬性。這將作為輸入傳遞給微調作業。在 AzureML Studio 模型目錄的模型詳細資訊頁面中，這也可作為資產 ID 欄位使用。

這個 Python 程式碼正在與 Azure Machine Learning (Azure ML) 服務互動。以下是它的功能分解：

它將 model_name 設定為「Phi-3-mini-4k-instruct」。

它使用 registry_ml_client 物件的 models 屬性的 get 方法，從 Azure ML 註冊表中檢索具有指定名稱的模型的最新版本。get 方法使用兩個參數呼叫：模型的名稱和指定應檢索模型最新版本的標籤。

它會在控制台列印一條訊息，指示將用於微調的模型的名稱、版本和 id。字串的 format 方法用於將模型的名稱、版本和 id 插入訊息中。模型的名稱、版本和 id 作為 foundation_model 物件的屬性來訪問。

```
# 設定 model 名稱
model_name = "Phi-3-mini-4k-instruct"

# 從 Azure ML registry 獲取最新版本的 model
foundation_model = registry_ml_client.models.get(model_name, label="latest")

# 列印 model 名稱、版本和 id
# 這些資訊對於追蹤和除錯很有用
print(
    "\n\n使用 model 名稱: {0}, 版本: {1}, id: {2} 進行微調".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)
```

## 建立一個計算資源以供工作使用

The finetune job works ONLY with GPU compute. The size of the compute depends on how big the model is and in most cases it becomes tricky to identify the right compute for the job. In this cell, we guide the user to select the right compute for the job.

finetune 工作僅適用於 GPU 計算。計算的大小取決於模型的大小，在大多數情況下，確定適合的計算變得很棘手。在此單元中，我們指導使用者選擇適合的計算。

**注意1** 下列計算機使用最優化的配置。任何配置的更改可能會導致 Cuda 記憶體不足錯誤。在這種情況下，請嘗試升級到更大的計算機規格。

**注意2** 在選擇 compute_cluster_size 時，請確保該計算資源在您的資源群組中可用。如果某個特定的計算資源不可用，您可以請求獲取該計算資源的訪問權限。

### 檢查模型是否支援微調

這個 Python 腳本正在與 Azure Machine Learning (Azure ML) 模型互動。以下是它的功能分解：

它匯入了 ast 模組，該模組提供處理 Python 抽象語法樹的函式。

它檢查 foundation_model 物件（代表 Azure ML 中的模型）是否具有名為 finetune_compute_allow_list 的標籤。Azure ML 中的標籤是可以建立並用來篩選和排序模型的鍵值對。

如果 finetune_compute_allow_list 標籤存在，它會使用 ast.literal_eval 函式來安全地解析標籤的值（字串）為一個 Python 清單。這個清單然後被指派給 computes_allow_list 變數。接著它會印出一個訊息，指示應該從清單中建立一個計算。

如果 finetune_compute_allow_list 標籤不存在，則將 computes_allow_list 設置為 None 並顯示一條訊息，指示 finetune_compute_allow_list 標籤不屬於模型的標籤。

總結來說，這個腳本正在檢查模型的 Metadata 中的特定標籤，如果標籤的值存在，則將其轉換為列表，並相應地向使用者提供反饋。

```
# 匯入 ast 模組，該模組提供處理 Python 抽象語法樹的函式
import ast

# 檢查模型的標籤中是否存在 'finetune_compute_allow_list' 標籤
if "finetune_compute_allow_list" in foundation_model.tags:
    # 如果標籤存在，使用 ast.literal_eval 安全地將標籤的值（一個字串）解析為 Python 清單
    computes_allow_list = ast.literal_eval(
        foundation_model.tags["finetune_compute_allow_list"]
    )  # 將字串轉換為 python 清單
    # 列印訊息，指示應從清單中建立計算
    print(f"請從上述清單中建立計算 - {computes_allow_list}")
else:
    # 如果標籤不存在，將 computes_allow_list 設為 None
    computes_allow_list = None
    # 列印訊息，指示 'finetune_compute_allow_list' 標籤不在模型的標籤中
    print("`finetune_compute_allow_list` 不在模型標籤中")
```

### 檢查 Compute Instance

這個 Python 腳本正在與 Azure Machine Learning (Azure ML) 服務互動，並對計算實例進行多項檢查。以下是它的功能分解：

它嘗試從 Azure ML 工作區中檢索儲存在 compute_cluster 中的計算實例。如果計算實例的佈建狀態為「失敗」，則引發 ValueError。

它檢查 computes_allow_list 是否不為 None。如果不是，它將列表中的所有 compute 大小轉換為小寫，並檢查當前 compute 實例的大小是否在列表中。如果不是，它會引發 ValueError。

如果 computes_allow_list 是 None ，它會檢查計算實例的大小是否在不支援的 GPU VM 大小列表中。如果是，則引發 ValueError 。

它檢索工作區中所有可用計算大小的列表。然後，它遍歷此列表，並對於每個計算大小，它檢查其名稱是否與當前計算實例的大小匹配。如果匹配，它檢索該計算大小的 GPU 數量並將 gpu_count_found 設置為 True。

如果 gpu_count_found 是 True，它會列印計算實例中的 GPU 數量。如果 gpu_count_found 是 False，它會引發 ValueError。

總結來說，這個腳本正在對 Azure ML 工作區中的計算實例進行多項檢查，包括檢查其佈建狀態、其大小是否在允許清單或拒絕清單中，以及它擁有的 GPU 數量。

```
# 列印例外訊息
print(e)
# 如果工作區中沒有可用的計算大小，則引發 ValueError
raise ValueError(
    f"警告！計算大小 {compute_cluster_size} 在工作區中不可用"
)

# 從 Azure ML 工作區檢索計算實例
compute = workspace_ml_client.compute.get(compute_cluster)
# 檢查計算實例的配置狀態是否為「失敗」
if compute.provisioning_state.lower() == "failed":
    # 如果配置狀態為「失敗」，則引發 ValueError
    raise ValueError(
        f"配置失敗，計算 '{compute_cluster}' 處於失敗狀態。"
        f"請嘗試建立不同的計算"
    )

# 檢查 computes_allow_list 是否不為 None
if computes_allow_list is not None:
    # 將 computes_allow_list 中的所有計算大小轉換為小寫
    computes_allow_list_lower_case = [x.lower() for x in computes_allow_list]
    # 檢查計算實例的大小是否在 computes_allow_list_lower_case 中
    if compute.size.lower() not in computes_allow_list_lower_case:
        # 如果計算實例的大小不在 computes_allow_list_lower_case 中，則引發 ValueError
        raise ValueError(
            f"VM 大小 {compute.size} 不在允許的計算列表中進行微調"
        )
else:
    # 定義不支援的 GPU VM 大小列表
    unsupported_gpu_vm_list = [
        "standard_nc6",
        "standard_nc12",
        "standard_nc24",
        "standard_nc24r",
    ]
    # 檢查計算實例的大小是否在 unsupported_gpu_vm_list 中
    if compute.size.lower() in unsupported_gpu_vm_list:
        # 如果計算實例的大小在 unsupported_gpu_vm_list 中，則引發 ValueError
        raise ValueError(
            f"VM 大小 {compute.size} 目前不支援進行微調"
        )

# 初始化一個標誌來檢查是否已找到計算實例中的 GPU 數量
gpu_count_found = False
# 檢索工作區中所有可用計算大小的列表
workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
available_sku_sizes = []
# 遍歷可用計算大小的列表
for compute_sku in workspace_compute_sku_list:
    available_sku_sizes.append(compute_sku.name)
    # 檢查計算大小的名稱是否與計算實例的大小匹配
    if compute_sku.name.lower() == compute.size.lower():
        # 如果匹配，則檢索該計算大小的 GPU 數量並將 gpu_count_found 設置為 True
        gpus_per_node = compute_sku.gpus
        gpu_count_found = True
# 如果 gpu_count_found 為 True，則列印計算實例中的 GPU 數量
if gpu_count_found:
    print(f"計算 {compute.size} 中的 GPU 數量：{gpus_per_node}")
else:
    # 如果 gpu_count_found 為 False，則引發 ValueError
    raise ValueError(
        f"未找到計算 {compute.size} 中的 GPU 數量。可用的 SKU 有：{available_sku_sizes}。"
        f"這不應該發生。請檢查所選的計算叢集：{compute_cluster} 並重試。"
    )
```

## 4. 選擇資料集以微調模型

我們使用 ultrachat_200k 資料集。該資料集有四個分割，適用於：

監督式微調 (sft)。
生成排名 (gen)。每個分割的例子數量如下所示：
train_sft	test_sft	train_gen	test_gen
207865	23110	256032	28304
接下來的幾個單元格顯示了微調的基本資料準備：

Visualize some data rows
我們希望這個範例能夠快速執行，所以保存包含已經修剪過的 5% 資料列的 train_sft 和 test_sft 檔案。這意味著微調後的模型將具有較低的準確性，因此不應該用於實際應用。
download-dataset.py 用於下載 ultrachat_200k 資料集並將資料集轉換為微調管道元件可消耗的格式。由於資料集很大，因此我們這裡只有部分資料集。

執行以下腳本僅下載 5% 的資料。這可以通過將 dataset_split_pc 參數更改為所需的百分比來增加。

**注意：** 一些語言模型有不同的語言程式碼，因此數據集中的欄位名稱應該反映相同的內容。

這是一個資料應該如何顯示的範例
聊天完成資料集以 parquet 格式儲存，每個項目使用以下結構:

這是一個 JSON（JavaScript 物件表示法）文件，這是一種流行的資料交換格式。它不是可執行的程式碼，而是一種儲存和傳輸資料的方式。以下是其結構的分解：

"prompt"：此鍵包含一個字串值，表示對 AI 助理提出的任務或問題。

"messages": 此鍵包含一個物件陣列。每個物件代表使用者與 AI 助手之間對話中的一則訊息。每個訊息物件有兩個鍵：

"content": 此鍵包含一個字串值，表示訊息的內容。
"role": 此鍵包含一個字串值，表示發送訊息的實體角色。它可以是 "user" 或 "assistant"。
"prompt_id": 此鍵包含一個字串值，表示提示的唯一標識符。

在這個特定的 JSON 文件中，表示了一個對話，其中使用者要求 AI 助手為反烏托邦故事建立一個主角。助手回應後，使用者接著要求更多細節。助手同意提供更多細節。整個對話與一個特定的 prompt id 相關聯。

```
{
    // 提供給 AI 助理的任務或問題
    "prompt": "建立一個完全發展的主角，他在暴君統治下的反烏托邦社會中面臨生存挑戰。...",

    // 一個物件陣列，每個物件代表使用者與 AI 助理之間對話中的一則訊息
    "messages":[
        {
            // 使用者訊息的內容
            "content": "建立一個完全發展的主角，他在暴君統治下的反烏托邦社會中面臨生存挑戰。...",
            // 發送訊息的實體角色
            "role": "user"
        },
        {
            // 助理訊息的內容
            "content": "名字: Ava\n\n Ava 當世界如她所知崩潰時，年僅 16 歲。政府倒台，留下了一個混亂無法無天的社會。...",
            // 發送訊息的實體角色
            "role": "assistant"
        },
        {
            // 使用者訊息的內容
            "content": "哇，Ava 的故事如此激烈且鼓舞人心！你能提供更多細節嗎。...",
            // 發送訊息的實體角色
            "role": "user"
        }, 
        {
            // 助理訊息的內容
            "content": "當然！....",
            // 發送訊息的實體角色
            "role": "assistant"
        }
    ],

    // 提示的唯一識別碼
    "prompt_id": "d938b65dfe31f05f80eb8572964c6673eddbd68eff3db6bd234d7f1e3b86c2af"
}
```

### 下載 資料

這個 Python 腳本用於使用名為 download-dataset.py 的輔助腳本下載數據集。以下是它的功能分解：

它匯入了 os 模組，這提供了一種使用作業系統相依功能的可攜式方式。

它使用 os.system 函式在 shell 中執行 download-dataset.py 腳本，並帶有特定的命令列參數。這些參數指定了要下載的資料集（HuggingFaceH4/ultrachat_200k）、下載到的目錄（ultrachat_200k_dataset）以及要分割的資料集百分比（5）。os.system 函式返回它執行的命令的退出狀態；這個狀態存儲在 exit_status 變數中。

它檢查 exit_status 是否不為 0。在類 Unix 作業系統中，exit status 為 0 通常表示命令成功，而任何其他數字表示錯誤。如果 exit_status 不為 0，它會引發一個 Exception，並顯示一條訊息，指示下載資料集時發生錯誤。

總結來說，這個腳本正在執行一個命令來使用輔助腳本下載數據集，如果命令失敗則會引發異常。

```
# 匯入 os 模組，這個模組提供使用作業系統相依功能的方法
import os

# 使用 os.system 函式在 shell 中執行 download-dataset.py 腳本，並帶有特定的命令列參數
# 這些參數指定要下載的資料集（HuggingFaceH4/ultrachat_200k）、要下載到的目錄（ultrachat_200k_dataset），以及要拆分的資料集百分比（5）
# os.system 函式返回它執行的命令的退出狀態；這個狀態被儲存在 exit_status 變數中
exit_status = os.system(
    "python ./download-dataset.py --dataset HuggingFaceH4/ultrachat_200k --download_dir ultrachat_200k_dataset --dataset_split_pc 5"
)

# 檢查 exit_status 是否不等於 0
# 在類 Unix 作業系統中，退出狀態為 0 通常表示命令成功，而任何其他數字表示錯誤
# 如果 exit_status 不等於 0，則引發一個 Exception，並帶有一條消息，指示下載資料集時發生錯誤
if exit_status != 0:
    raise Exception("Error downloading dataset")
```

### 將資料載入 DataFrame

這個 Python 程式正在將一個 JSON Lines 檔案載入 pandas DataFrame 並顯示前 5 行。以下是它的功能分解：

它匯入 pandas 函式庫，這是一個強大的資料操作和分析函式庫。

它將 pandas 的顯示選項的最大欄寬設置為 0。這意味著當 DataFrame 被列印時，每一欄的完整文字將會顯示而不會被截斷。

它使用 pd.read_json 函式從 ultrachat_200k_dataset 目錄中載入 train_sft.jsonl 檔案到 DataFrame。lines=True 參數表示檔案是 JSON Lines 格式，其中每行是一個獨立的 JSON 物件。

它使用 head 方法來顯示 DataFrame 的前 5 行。如果 DataFrame 少於 5 行，它將顯示所有行。

總結來說，這個腳本正在將一個 JSON Lines 檔案載入到一個 DataFrame 並顯示前 5 行的完整欄位文字。

```
# 匯入 pandas 函式庫，這是一個強大的資料操作和分析函式庫
import pandas as pd

# 設定 pandas 的顯示選項中的最大欄位寬度為 0
# 這意味著當 DataFrame 被列印時，每個欄位的完整文字將會顯示而不會被截斷
pd.set_option("display.max_colwidth", 0)

# 使用 pd.read_json 函式從 ultrachat_200k_dataset 目錄中載入 train_sft.jsonl 檔案到 DataFrame
# lines=True 參數表示該檔案是 JSON Lines 格式，每一行是一個獨立的 JSON 物件
df = pd.read_json("./ultrachat_200k_dataset/train_sft.jsonl", lines=True)

# 使用 head 方法顯示 DataFrame 的前 5 行
# 如果 DataFrame 少於 5 行，它將顯示所有行
df.head()
```

## 提交微調工作，使用模型和資料作為輸入

建立使用 chat-completion pipeline 元件的工作。了解更多關於微調所支援的所有參數。

### 定義 finetune 參數

Finetune 參數可以分為兩類 - 訓練參數、優化參數

Training 參數定義了訓練方面，例如 -

- 優化器、排程器的使用
- 優化微調的指標
- 訓練步驟數量和批次大小等
- 優化參數有助於優化 GPU 記憶體並有效利用計算資源。

以下是屬於此類別的一些參數。最佳化參數因每個模型而異，並與模型一起打包以處理這些變化。

- 啟用 deepspeed 和 LoRA
- 啟用混合精度訓練
- 啟用多節點訓練

**注意：** 監督微調可能會導致對齊丟失或災難性遺忘。我們建議在微調後檢查此問題並執行對齊階段。

### 調整參數

這個 Python 程式碼正在設定參數以微調機器學習模型。以下是它的功能分解：

它設定了預設的訓練參數，例如訓練週期數、訓練和評估的批次大小、學習率和學習率調度器類型。

它設定了預設的優化參數，例如是否應用 Layer-wise Relevance Propagation（LoRa）和 DeepSpeed，以及 DeepSpeed 階段。

它將訓練和優化參數結合到一個稱為 finetune_parameters 的單一字典中。

它會檢查 foundation_model 是否有任何模型特定的預設參數。如果有，它會打印一條警告訊息並使用這些模型特定的預設值更新 finetune_parameters 字典。ast.literal_eval 函式被用來將模型特定的預設值從字串轉換為 Python 字典。

它會列印將用於執行的最終微調參數集。

總結來說，這個腳本正在設定和顯示用於微調機器學習模型的參數，並且能夠使用模型特定的參數來覆蓋預設參數。

```
# 設定預設的訓練參數，例如訓練週期數、訓練和評估的批次大小、學習率和學習率調度器類型
training_parameters = dict(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
)

# 設定預設的優化參數，例如是否應用層級相關傳播（LoRa）和 DeepSpeed，以及 DeepSpeed 階段
optimization_parameters = dict(
    apply_lora="true",
    apply_deepspeed="true",
    deepspeed_stage=2,
)

# 將訓練和優化參數合併到一個名為 finetune_parameters 的單一字典中
finetune_parameters = {**training_parameters, **optimization_parameters}

# 檢查 foundation_model 是否有任何模型特定的預設參數
# 如果有，則打印警告訊息並使用這些模型特定的預設來更新 finetune_parameters 字典
# ast.literal_eval 函式用於將模型特定的預設從字串轉換為 Python 字典
if "model_specific_defaults" in foundation_model.tags:
    print("Warning! Model specific defaults exist. The defaults could be overridden.")
    finetune_parameters.update(
        ast.literal_eval(  # 將字串轉換為 python 字典
            foundation_model.tags["model_specific_defaults"]
        )
    )

# 打印將用於執行的最終微調參數集
print(
    f"The following finetune parameters are going to be set for the run: {finetune_parameters}"
)
```

### 訓練管道

這個 Python 程式碼定義了一個函式來產生機器學習訓練管道的顯示名稱，然後呼叫這個函式來產生並列印顯示名稱。以下是它的作用分解：

The get_pipeline_display_name 函式被定義。這個函式根據與訓練管道相關的各種參數產生顯示名稱。

Inside the 函式, it calculates the total batch size by multiplying the per-device batch size, the number of gradient accumulation steps, the number of GPUs per node, and the number of nodes used for fine-tuning.

它檢索各種其他參數，例如學習率調度器類型、是否應用 DeepSpeed、DeepSpeed 階段、是否應用 Layer-wise Relevance Propagation (LoRa)、保留的模型檢查點數量限制以及最大序列長度。

它建構了一個包含所有這些參數的字串，並以連字號分隔。如果應用了 DeepSpeed 或 LoRa，則字串分別包含 "ds" 後接 DeepSpeed 階段，或 "lora"。如果沒有，則分別包含 "nods" 或 "nolora"。

函式返回此字串，作為訓練管道的顯示名稱。

在 函式 定義 後，會 呼叫 它 來 產生 顯示 名稱，然後 將其 列印。

總結來說，這個腳本根據各種參數生成一個機器學習訓練管道的顯示名稱，然後打印這個顯示名稱。

```
# 定義一個函式來產生訓練管道的顯示名稱
def get_pipeline_display_name():
    # 通過將每個裝置的批次大小、梯度累積步數、每個節點的 GPU 數量和用於微調的節點數量相乘來計算總批次大小
    batch_size = (
        int(finetune_parameters.get("per_device_train_batch_size", 1))
        * int(finetune_parameters.get("gradient_accumulation_steps", 1))
        * int(gpus_per_node)
        * int(finetune_parameters.get("num_nodes_finetune", 1))
    )
    # 獲取學習率調度器類型
    scheduler = finetune_parameters.get("lr_scheduler_type", "linear")
    # 獲取是否應用了 DeepSpeed
    deepspeed = finetune_parameters.get("apply_deepspeed", "false")
    # 獲取 DeepSpeed 階段
    ds_stage = finetune_parameters.get("deepspeed_stage", "2")
    # 如果應用了 DeepSpeed，則在顯示名稱中包含 "ds" 和 DeepSpeed 階段；如果沒有，則包含 "nods"
    if deepspeed == "true":
        ds_string = f"ds{ds_stage}"
    else:
        ds_string = "nods"
    # 獲取是否應用了層級相關傳播 (LoRa)
    lora = finetune_parameters.get("apply_lora", "false")
    # 如果應用了 LoRa，則在顯示名稱中包含 "lora"；如果沒有，則包含 "nolora"
    if lora == "true":
        lora_string = "lora"
    else:
        lora_string = "nolora"
    # 獲取要保留的模型檢查點數量限制
    save_limit = finetune_parameters.get("save_total_limit", -1)
    # 獲取最大序列長度
    seq_len = finetune_parameters.get("max_seq_length", -1)
    # 通過將所有這些參數用連字符連接來構建顯示名稱
    return (
        model_name
        + "-"
        + "ultrachat"
        + "-"
        + f"bs{batch_size}"
        + "-"
        + f"{scheduler}"
        + "-"
        + ds_string
        + "-"
        + lora_string
        + f"-save_limit{save_limit}"
        + f"-seqlen{seq_len}"
    )

# 呼叫函式來產生顯示名稱
pipeline_display_name = get_pipeline_display_name()
# 列印顯示名稱
print(f"Display name used for the run: {pipeline_display_name}")
```

### 設定 Pipeline

這個 Python 腳本正在使用 Azure Machine Learning SDK 定義和配置一個機器學習管道。以下是它的作用分解：

1. 它從 Azure AI ML SDK 匯入必要的模組。

2. 它從註冊表中獲取名為 "chat_completion_pipeline" 的管道元件。

3. 它使用 `@pipeline` 裝飾器和函式 `create_pipeline` 定義一個管道工作。管道的名稱設置為 `pipeline_display_name`。

4. 在 `create_pipeline` 函式內，它使用各種參數初始化獲取的管道元件，包括模型路徑、不同階段的計算叢集、訓練和測試的數據集拆分、用於微調的 GPU 數量以及其他微調參數。

5. 它將微調工作的輸出映射到管道工作的輸出。這樣做是為了便於註冊微調模型，這是將模型部署到線上或批次端點所需的。

6. 它通過呼叫 `create_pipeline` 函式建立管道的實例。

7. 它將管道的 `force_rerun` 設定設置為 `True`，這意味著不會使用先前工作的快取結果。

8. 它將管道的 `continue_on_step_failure` 設定設置為 `False`，這意味著如果任何步驟失敗，管道將停止。

總結來說，這個腳本正在使用 Azure Machine Learning SDK 定義和配置一個用於聊天完成任務的機器學習管道。

```
# Import necessary modules from the Azure AI ML SDK
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input

# Fetch the pipeline component named "chat_completion_pipeline" from the registry
pipeline_component_func = registry_ml_client.components.get(
    name="chat_completion_pipeline", label="latest"
)

# Define the pipeline job using the @pipeline decorator and the function create_pipeline
# The name of the pipeline is set to pipeline_display_name
@pipeline(name=pipeline_display_name)
def create_pipeline():
    # Initialize the fetched pipeline component with various parameters
    # These include the model path, compute clusters for different stages, dataset splits for training and testing, the number of GPUs to use for fine-tuning, and other fine-tuning parameters
    chat_completion_pipeline = pipeline_component_func(
        mlflow_model_path=foundation_model.id,
        compute_model_import=compute_cluster,
        compute_preprocess=compute_cluster,
        compute_finetune=compute_cluster,
        compute_model_evaluation=compute_cluster,
        # Map the dataset splits to parameters
        train_file_path=Input(
            type="uri_file", path="./ultrachat_200k_dataset/train_sft.jsonl"
        ),
        test_file_path=Input(
            type="uri_file", path="./ultrachat_200k_dataset/test_sft.jsonl"
        ),
        # Training settings
        number_of_gpu_to_use_finetuning=gpus_per_node,  # Set to the number of GPUs available in the compute
        **finetune_parameters
    )
    return {
        # Map the output of the fine tuning job to the output of pipeline job
        # This is done so that we can easily register the fine tuned model
        # Registering the model is required to deploy the model to an online or batch endpoint
        "trained_model": chat_completion_pipeline.outputs.mlflow_model_folder
    }

# Create an instance of the pipeline by calling the create_pipeline function
pipeline_object = create_pipeline()

# Don't use cached results from previous jobs
pipeline_object.settings.force_rerun = True

# Set continue on step failure to False
# This means that the pipeline will stop if any step fails
pipeline_object.settings.continue_on_step_failure = False
```

### 提交工作

這個 Python 腳本正在提交一個機器學習管道工作到 Azure 機器學習工作區，然後等待工作完成。以下是它的操作分解：

它呼叫 workspace_ml_client 中 jobs 物件的 create_or_update 方法來提交 pipeline 工作。要執行的 pipeline 由 pipeline_object 指定，而工作所屬的實驗由 experiment_name 指定。

它然後呼叫 workspace_ml_client 中 jobs 物件的 stream 方法來等待 pipeline 工作完成。要等待的工作由 pipeline_job 物件的 name 屬性指定。

總結來說，這個腳本正在提交一個機器學習管道工作到 Azure 機器學習工作區，然後等待工作完成。

```
# 提交 pipeline 工作到 Azure Machine Learning 工作區
# 要執行的 pipeline 由 pipeline_object 指定
# 工作所屬的實驗由 experiment_name 指定
pipeline_job = workspace_ml_client.jobs.create_or_update(
    pipeline_object, experiment_name=experiment_name
)

# 等待 pipeline 工作完成
# 要等待的工作由 pipeline_job 物件的 name 屬性指定
workspace_ml_client.jobs.stream(pipeline_job.name)
```

## 6. 註冊微調後的模型到工作區

我們將從微調工作的輸出中註冊模型。這將追蹤微調模型與微調工作之間的譜系。進一步地，微調工作會追蹤到基礎模型、數據和訓練程式碼的譜系。

### 註冊 ML 模型

這個 Python 腳本正在註冊一個在 Azure Machine Learning 管線中訓練的機器學習模型。以下是它的功能分解：

它從 Azure AI ML SDK 匯入必要的模組。

它透過呼叫 workspace_ml_client 中 jobs 物件的 get 方法並存取其 outputs 屬性，檢查 trained_model 輸出是否可從 pipeline 作業中取得。

它透過格式化字串來建構通往訓練模型的路徑，使用管道工作的名稱和輸出（"trained_model"）的名稱。

它通過將「-ultrachat-200k」附加到原始模型名稱並將任何斜線替換為連字符來定義微調模型的名稱。

它準備透過建立一個 Model 物件來註冊模型，包含各種參數，包括模型的路徑、模型的類型（MLflow 模型）、模型的名稱和版本，以及模型的描述。

它透過呼叫 workspace_ml_client 中 models 物件的 create_or_update 方法，並以 Model 物件作為參數來註冊模型。

它會列印已註冊的模型。

總結來說，這個腳本正在註冊一個在 Azure 機器學習管道中訓練的機器學習模型。

```
# Import necessary modules from the Azure AI ML SDK
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# Check if the `trained_model` output is available from the pipeline job
print("pipeline job outputs: ", workspace_ml_client.jobs.get(pipeline_job.name).outputs)

# Construct a path to the trained model by formatting a string with the name of the pipeline job and the name of the output ("trained_model")
model_path_from_job = "azureml://jobs/{0}/outputs/{1}".format(
    pipeline_job.name, "trained_model"
)

# Define a name for the fine-tuned model by appending "-ultrachat-200k" to the original model name and replacing any slashes with hyphens
finetuned_model_name = model_name + "-ultrachat-200k"
finetuned_model_name = finetuned_model_name.replace("/", "-")

print("path to register model: ", model_path_from_job)

# Prepare to register the model by creating a Model object with various parameters
# These include the path to the model, the type of the model (MLflow model), the name and version of the model, and a description of the model
prepare_to_register_model = Model(
    path=model_path_from_job,
    type=AssetTypes.MLFLOW_MODEL,
    name=finetuned_model_name,
    version=timestamp,  # Use timestamp as version to avoid version conflict
    description=model_name + " fine tuned model for ultrachat 200k chat-completion",
)

print("prepare to register model: \n", prepare_to_register_model)

# Register the model by calling the create_or_update method of the models object in the workspace_ml_client with the Model object as the argument
registered_model = workspace_ml_client.models.create_or_update(
    prepare_to_register_model
)

# Print the registered model
print("registered model: \n", registered_model)
```

## 部署微調後的模型到線上端點

Online endpoints 提供了一個持久的 REST API，可以用來整合需要使用模型的應用程式。

### 管理端點

這個 Python 腳本正在 Azure Machine Learning 中為已註冊的模型建立一個受管理的線上端點。以下是它的功能分解：

它從 Azure AI ML SDK 匯入必要的模組。

它通過將時間戳附加到字串 "ultrachat-completion-" 來定義線上端點的唯一名稱。

它準備透過建立一個 ManagedOnlineEndpoint 物件來建立線上端點，包含各種參數，包括端點的名稱、端點的描述和驗證模式（"key"）。

它透過呼叫 workspace_ml_client 的 begin_create_or_update 方法並將 ManagedOnlineEndpoint 物件作為參數來建立線上端點。然後透過呼叫 wait 方法等待建立操作完成。

總結來說，這個腳本是在 Azure Machine Learning 中為註冊的模型建立一個受管理的線上端點。

```
# Import necessary modules from the Azure AI ML SDK
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    ProbeSettings,
    OnlineRequestSettings,
)

# Define a unique name for the online endpoint by appending a timestamp to the string "ultrachat-completion-"
online_endpoint_name = "ultrachat-completion-" + timestamp

# Prepare to create the online endpoint by creating a ManagedOnlineEndpoint object with various parameters
# These include the name of the endpoint, a description of the endpoint, and the authentication mode ("key")
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for "
    + registered_model.name
    + ", fine tuned model for ultrachat-200k-chat-completion",
    auth_mode="key",
)

# Create the online endpoint by calling the begin_create_or_update method of the workspace_ml_client with the ManagedOnlineEndpoint object as the argument
# Then wait for the creation operation to complete by calling the wait method
workspace_ml_client.begin_create_or_update(endpoint).wait()
```

您可以在此找到支援部署的 SKU 清單 - [Managed online endpoints SKU list](https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list)

### 部署 ML 模型

這個 Python 腳本正在將註冊的機器學習模型部署到 Azure Machine Learning 中的受管線上端點。以下是它的功能分解：

它匯入了 ast 模組，該模組提供了處理 Python 抽象語法樹的函式。

它將部署的實例類型設置為 "Standard_NC6s_v3"。

它檢查 foundation model 中是否存在 inference_compute_allow_list 標籤。如果存在，它會將標籤值從字串轉換為 Python 清單並將其指派給 inference_computes_allow_list。如果不存在，它會將 inference_computes_allow_list 設定為 None。

它檢查指定的 instance 類型是否在允許清單中。如果不在，它會顯示一條訊息，要求使用者從允許清單中選擇一個 instance 類型。

它準備透過建立一個 ManagedOnlineDeployment 物件來建立部署，包含各種參數，包括部署的名稱、端點的名稱、模型的 ID、實例類型和數量、存活探測設定以及請求設定。

它透過呼叫 workspace_ml_client 的 begin_create_or_update 方法並以 ManagedOnlineDeployment 物件作為參數來建立部署。然後透過呼叫 wait 方法等待建立操作完成。

它將端點的流量設定為將 100% 的流量直接導向 "demo" 部署。

它透過呼叫 workspace_ml_client 的 begin_create_or_update 方法並將 endpoint 物件作為參數來更新 endpoint。然後透過呼叫 result 方法等待更新操作完成。

總結來說，這個腳本正在將註冊的機器學習模型部署到 Azure 機器學習中的受管線上端點。

```
# 匯入 ast 模組，它提供處理 Python 抽象語法樹的函式
import ast

# 設定部署的實例類型
instance_type = "Standard_NC6s_v3"

# 檢查 foundation_model 中是否存在 `inference_compute_allow_list` 標籤
if "inference_compute_allow_list" in foundation_model.tags:
    # 如果存在，將標籤值從字串轉換為 Python 清單並賦值給 `inference_computes_allow_list`
    inference_computes_allow_list = ast.literal_eval(
        foundation_model.tags["inference_compute_allow_list"]
    )
    print(f"請從上述清單中建立一個計算資源 - {computes_allow_list}")
else:
    # 如果不存在，將 `inference_computes_allow_list` 設為 `None`
    inference_computes_allow_list = None
    print("`inference_compute_allow_list` 不在模型標籤中")

# 檢查指定的實例類型是否在允許清單中
if (
    inference_computes_allow_list is not None
    and instance_type not in inference_computes_allow_list
):
    print(
        f"`instance_type` 不在允許的計算資源中。請從 {inference_computes_allow_list} 中選擇一個值"
    )

# 準備建立部署，通過建立一個帶有各種參數的 `ManagedOnlineDeployment` 物件
demo_deployment = ManagedOnlineDeployment(
    name="demo",
    endpoint_name=online_endpoint_name,
    model=registered_model.id,
    instance_type=instance_type,
    instance_count=1,
    liveness_probe=ProbeSettings(initial_delay=600),
    request_settings=OnlineRequestSettings(request_timeout_ms=90000),
)

# 通過呼叫 `workspace_ml_client` 的 `begin_create_or_update` 方法並將 `ManagedOnlineDeployment` 物件作為參數來建立部署
# 然後通過呼叫 `wait` 方法等待建立操作完成
workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()

# 設定端點的流量，將 100% 的流量導向 "demo" 部署
endpoint.traffic = {"demo": 100}

# 通過呼叫 `workspace_ml_client` 的 `begin_create_or_update` 方法並將 `endpoint` 物件作為參數來更新端點
# 然後通過呼叫 `result` 方法等待更新操作完成
workspace_ml_client.begin_create_or_update(endpoint).result()
```

## 測試端點與範例資料

我們將從測試資料集中獲取一些範例資料並提交到線上端點進行推論。然後我們將顯示得分標籤與真實標籤並排顯示

### 閱讀結果

這個 Python 程式正在讀取一個 JSON Lines 檔案到一個 pandas DataFrame，取一個隨機樣本，並重新設定索引。以下是它的功能分解：

它讀取檔案 ./ultrachat_200k_dataset/test_gen.jsonl 到一個 pandas DataFrame。因為檔案是 JSON Lines 格式，每一行都是一個單獨的 JSON 物件，所以使用 read_json 函式並帶有 lines=True 參數。

它從 DataFrame 中隨機抽取 1 行。sample 函式與 n=1 參數一起使用，以指定要選擇的隨機行數。

它會重設 DataFrame 的索引。reset_index 函式與 drop=True 參數一起使用，以刪除原始索引並用新的預設整數值索引替換。

它使用 head 函式和參數 2 顯示 DataFrame 的前 2 行。然而，由於 DataFrame 在抽樣後只包含一行，這將只顯示那一行。

總結來說，這個腳本正在將 JSON Lines 檔案讀取到 pandas DataFrame 中，隨機取樣 1 行，重設索引，並顯示第一行。

```
# 匯入 pandas 函式庫
import pandas as pd

# 將 JSON Lines 檔案 './ultrachat_200k_dataset/test_gen.jsonl' 讀取到 pandas DataFrame 中
# 'lines=True' 參數表示該檔案為 JSON Lines 格式，每一行是一個獨立的 JSON 物件
test_df = pd.read_json("./ultrachat_200k_dataset/test_gen.jsonl", lines=True)

# 從 DataFrame 中隨機取樣 1 行
# 'n=1' 參數指定要選擇的隨機行數
test_df = test_df.sample(n=1)

# 重設 DataFrame 的索引
# 'drop=True' 參數表示應該丟棄原始索引並用新的預設整數值索引取代
# 'inplace=True' 參數表示應該就地修改 DataFrame（不建立新的物件）
test_df.reset_index(drop=True, inplace=True)

# 顯示 DataFrame 的前 2 行
# 然而，由於取樣後 DataFrame 只包含一行，這將只顯示那一行
test_df.head(2)
```

### 建立 JSON 物件

這個 Python 腳本正在建立一個具有特定參數的 JSON 物件並將其儲存到檔案中。以下是它的功能分解：

它匯入了 json 模組，該模組提供了處理 JSON 資料的函式。

它建立一個字典參數，其中的鍵和值代表機器學習模型的參數。鍵是 "temperature"、"top_p"、"do_sample" 和 "max_new_tokens"，對應的值分別是 0.6、0.9、True 和 200。

它建立另一個字典 test_json，包含兩個鍵：「input_data」和「params」。其中「input_data」的值是另一個字典，包含鍵「input_string」和「parameters」。其中「input_string」的值是一個列表，包含來自 test_df DataFrame 的第一條訊息。「parameters」的值是先前建立的參數字典。「params」的值是一個空字典。

它打開了一個名為 sample_score.json 的檔案

```
# 匯入 json 模組，它提供了處理 JSON 資料的函式
import json

# 建立一個字典 `parameters`，其鍵和值代表機器學習模型的參數
# 鍵是 "temperature"、"top_p"、"do_sample" 和 "max_new_tokens"，對應的值分別是 0.6、0.9、True 和 200
parameters = {
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "max_new_tokens": 200,
}

# 建立另一個字典 `test_json`，它有兩個鍵："input_data" 和 "params"
# "input_data" 的值是另一個字典，包含鍵 "input_string" 和 "parameters"
# "input_string" 的值是一個列表，包含來自 `test_df` DataFrame 的第一條訊息
# "parameters" 的值是之前建立的 `parameters` 字典
# "params" 的值是一個空字典
test_json = {
    "input_data": {
        "input_string": [test_df["messages"][0]],
        "parameters": parameters,
    },
    "params": {},
}

# 以寫入模式打開位於 `./ultrachat_200k_dataset` 目錄中的名為 `sample_score.json` 的檔案
with open("./ultrachat_200k_dataset/sample_score.json", "w") as f:
    # 使用 `json.dump` 函式將 `test_json` 字典以 JSON 格式寫入檔案
    json.dump(test_json, f)
```

### 呼叫端點

這個 Python 程式碼正在呼叫 Azure Machine Learning 中的線上端點來評分一個 JSON 檔案。以下是它的功能分解：

它呼叫 workspace_ml_client 物件的 online_endpoints 屬性的 invoke 方法。此方法用於向線上端點發送請求並獲取回應。

它指定端點的名稱和部署，使用 endpoint_name 和 deployment_name 參數。在這種情況下，端點名稱存儲在 online_endpoint_name 變數中，部署名稱是 "demo"。

它指定了使用 request_file 參數的 JSON 檔案路徑。在這個例子中，檔案是 ./ultrachat_200k_dataset/sample_score.json。

它將來自端點的回應儲存在 response 變數中。

它列印原始回應。

總結來說，這個腳本正在呼叫 Azure Machine Learning 的線上端點來評分 JSON 檔案並列印回應。

```
# 呼叫 Azure Machine Learning 的線上端點來評分 `sample_score.json` 檔案
# 使用 `workspace_ml_client` 物件的 `online_endpoints` 屬性的 `invoke` 方法來發送請求到線上端點並獲取回應
# `endpoint_name` 參數指定端點的名稱，該名稱存儲在 `online_endpoint_name` 變數中
# `deployment_name` 參數指定部署的名稱，這裡是 "demo"
# `request_file` 參數指定要評分的 JSON 檔案的路徑，即 `./ultrachat_200k_dataset/sample_score.json`
response = workspace_ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="demo",
    request_file="./ultrachat_200k_dataset/sample_score.json",
)

# 打印來自端點的原始回應
print("raw response: \n", response, "\n")
```

## 刪除線上端點

不要忘記刪除線上端點，否則您將會因端點使用的計算資源而持續計費。這行 Python 程式碼正在刪除 Azure Machine Learning 中的線上端點。以下是它的作用分解：

它呼叫 workspace_ml_client 物件的 online_endpoints 屬性的 begin_delete 方法。此方法用於開始刪除線上端點。

它使用 name 參數指定要刪除的 endpoint 名稱。在這種情況下，endpoint 名稱存儲在 online_endpoint_name 變數中。

它呼叫 wait 方法來等待刪除操作完成。這是一個阻塞操作，意味著它將阻止腳本繼續執行直到刪除完成。

總結來說，這行程式碼正在啟動刪除 Azure Machine Learning 中的線上端點，並等待操作完成。

```
# 刪除 Azure Machine Learning 中的線上端點
# `workspace_ml_client` 物件的 `online_endpoints` 屬性的 `begin_delete` 方法用於開始刪除線上端點
# `name` 參數指定要刪除的端點名稱，該名稱存儲在 `online_endpoint_name` 變數中
# 調用 `wait` 方法等待刪除操作完成。這是一個阻塞操作，意味著它會阻止腳本繼續執行直到刪除完成
workspace_ml_client.online_endpoints.begin_delete(name=online_endpoint_name).wait()
```

