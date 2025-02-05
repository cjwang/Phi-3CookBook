﻿# 透過從 Hugging Face 下載 DataSet 和相關圖片來生成圖像資料集

### 概述

這個腳本透過下載所需的圖像、過濾掉圖像下載失敗的列，並將資料集儲存為 CSV 檔案來準備機器學習的資料集。

### 先決條件

在 執行 這個 腳本 之前，請 確認 已 安裝 以下 函式庫：`Pandas`、`Datasets`、`requests`、`PIL` 和 `io`。你 也 需要 將 第 2 行 的 `'Insert_Your_Dataset'` 替換 為 你 從 Hugging Face 獲得 的 資料集 名稱。

必需的函式庫：

```python

import os
import pandas as pd
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
```

### 功能

The script performs the following steps:

1. 使用 `load_dataset()` 函式從 Hugging Face 下載資料集。
2. 使用 `to_pandas()` 方法將 Hugging Face 資料集轉換為 Pandas DataFrame 以便更容易操作。
3. 建立目錄以保存資料集和圖片。
4. 通過遍歷 DataFrame 中的每一行，使用自定義的 `download_image()` 函式下載圖片，並將過濾後的行附加到名為 `filtered_rows` 的新 DataFrame 中，過濾掉圖片下載失敗的行。
5. 建立一個包含過濾行的新 DataFrame，並將其保存到磁碟上作為 CSV 檔案。
6. 打印一條訊息，指示資料集和圖片已保存的位置。

### 自訂函式

`download_image()` 函式從 URL 下載圖片並使用 Pillow Image Library (PIL) 和 `io` 模組將其本地保存。如果圖片成功下載則返回 True，否則返回 False。當請求失敗時，該函式還會引發帶有錯誤訊息的例外。

### 這是如何運作的

下載圖像的函式 download_image 接受兩個參數：image_url（要下載的圖像的 URL）和 save_path（下載的圖像將被儲存的路徑）。

Here's how the 函式 works:

它開始通過使用 requests.get 方法向 image_url 發出 GET 請求。這會從 URL 檢索影像資料。

The `response.raise_for_status()` 行檢查請求是否成功。如果回應狀態碼顯示錯誤（例如，404 - Not Found），它將引發例外。這確保我們只有在請求成功時才繼續下載圖片。

圖像資料然後被傳遞給來自 PIL (Python Imaging Library) 模組的 Image.open 方法。此方法從圖像資料建立一個 Image 物件。

影像 .save(save_path) 行將影像儲存到指定的 save_path。save_path 應包括所需的檔案名稱和副檔名。

最後，該 函式 返回 True 以表示影像已成功下載並儲存。如果在此過程中發生任何例外狀況，它會捕捉例外狀況，列印一條指示失敗的錯誤訊息，並返回 False。

這個 函式 對於從 URL 下載圖片並將其本地儲存非常有用。它處理下載過程中可能出現的錯誤，並提供下載是否成功的反饋。

值得注意的是，requests 函式庫用於發送 HTTP 請求，PIL 函式庫用於處理圖像，而 BytesIO 類別用於將圖像資料作為位元組流來處理。

### 結論

這個程式碼提供了一種方便的方法來準備機器學習的資料集，通過下載所需的圖片，過濾掉圖片下載失敗的行，並將資料集儲存為 CSV 檔案。

### 範例腳本

```python
import os
import pandas as pd
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO

def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return False


# Download the dataset from Hugging Face
dataset = load_dataset('Insert_Your_Dataset')


# Convert the Hugging Face dataset to a Pandas DataFrame
df = dataset['train'].to_pandas()


# Create directories to save the dataset and images
dataset_dir = './data/DataSetName'
images_dir = os.path.join(dataset_dir, 'images')
os.makedirs(images_dir, exist_ok=True)


# Filter out rows where image download fails
filtered_rows = []
for idx, row in df.iterrows():
    image_url = row['imageurl']
    image_name = f"{row['product_code']}.jpg"
    image_path = os.path.join(images_dir, image_name)
    if download_image(image_url, image_path):
        row['local_image_path'] = image_path
        filtered_rows.append(row)


# Create a new DataFrame with the filtered rows
filtered_df = pd.DataFrame(filtered_rows)


# Save the updated dataset to disk
dataset_path = os.path.join(dataset_dir, 'Dataset.csv')
filtered_df.to_csv(dataset_path, index=False)


print(f"Dataset and images saved to {dataset_dir}")
```

### 範例程式碼下載

[產生 資料集](../../code/04.Finetuning/generate_dataset.py)

