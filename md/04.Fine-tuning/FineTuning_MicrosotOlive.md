# **微調 Phi-3 與 Microsoft Olive**

[Olive](https://github.com/microsoft/OLive?WT.mc_id=aiml-138114-kinfeylo) 是一個易於使用的硬體感知模型優化工具，結合了業界領先的模型壓縮、優化和編譯技術。

它的設計目的是簡化優化機器學習模型的過程，確保它們能夠最有效地利用特定硬體架構。

無論您是在雲端應用程式或邊緣設備上工作，Olive 都能讓您輕鬆且有效地優化您的模型。

## 主要功能:

- Olive 聚合並自動化針對目標硬體的最佳化技術。
- 沒有單一的最佳化技術適用於所有情境，因此 Olive 允許擴展性，讓業界專家能夠插入他們的最佳化創新。

## 減少工程工作量:

- 開發人員經常需要學習和使用多個特定硬體供應商的工具鏈來準備和優化訓練模型以進行部署。
- Olive 通過自動化優化技術來簡化這種體驗，以適應所需的硬體。

## 即用型 E2E 優化解決方案:

By composing and tuning integrated techniques, Olive offers a unified solution for end-to-end optimization.  
它在優化模型時考慮了準確性和延遲等限制條件。

## **使用 Microsoft Olive 進行微調**

Microsoft Olive 是一個非常易於使用的開源模型優化工具，可以涵蓋生成式人工智慧領域中的微調和參考。它只需要簡單的配置，結合使用開源的小語言模型和相關的執行環境（AzureML / local GPU, CPU, DirectML），就可以通過自動優化完成模型的微調或參考，並找到最佳模型部署到雲端或邊緣設備上。允許企業在本地和雲端建構自己的行業垂直模型。

![intro](../../imgs/04/02/intro.png)

## Phi-3 微調與 Microsoft Olive

![使用 Olive 進行微調](../../imgs/04/03/olivefinetune.png)

### **設定 Microsoft Olive**

Microsoft Olive 安裝非常簡單，並且也可以安裝於 CPU、GPU、DirectML 和 Azure ML

Setup Microsoft Olive
Microsoft Olive 安裝非常簡單，並且可以安裝在 CPU、GPU、DirectML 和 Azure ML

```bash

pip install olive-ai

```

如果你希望使用 CPU 執行一個 ONNX 模型，你可以使用

```bash

pip install olive-ai[cpu]

```

如果你想使用 GPU 執行一個 ONNX 模型，你可以使用

```bash

pip install olive-ai[gpu]

```

如果你想使用 Azure ML，請使用

pip install git+https://github.com/microsoft/Olive#egg=olive-ai[azureml]

***注意***

OS 要求：Ubuntu 20.04 / 22.04

### **Microsoft Olive 的 Config.json**

安裝後，您可以通過 Config 檔案設定不同模型特定的設定，包括資料、計算、訓練、部署和模型產生。

**1. 資料**

在 Microsoft Olive 上，可以支援本地資料和雲端資料的訓練，並且可以在設定中進行配置。

*本地 資料 設定*

您可以簡單地設定需要訓練以進行微調的資料集，通常是 json 格式，並使用資料模板進行調整。這需要根據模型的需求進行調整（例如，將其調整為 Microsoft Phi-3-mini 所需的格式。如果您有其他模型，請參考其他模型所需的微調格式進行處理）

```json

    "data_configs": {
        "dataset-default_train": {
            "name": "dataset-default",
            "type": "HuggingfaceContainer",
            "params_config": {
                "data_name": "json", 
                "data_files":"dataset/dataset-classification.json",
                "split": "train",
                "component_kwargs": {
                    "pre_process_data": {
                        "dataset_type": "corpus",
                        "text_cols": [
                            "phrase",
                            "tone"
                        ],
                        "text_template": "### 文字: {phrase}\n### 語氣是:\n{tone}",
                        "corpus_strategy": "join",
                        "source_max_len": 1024,
                        "pad_to_max_len": false,
                        "use_attention_mask": false
                    }
                }
            }
        }
    },

```

*雲端資料來源設定*

通過連結 Azure AI Studio/Azure Machine Learning Service 的資料存儲來連結雲端中的資料，您可以選擇通過 Microsoft Fabric 和 Azure Data 將不同的資料來源引入 Azure AI Studio/Azure Machine Learning Service，作為微調資料的支持。

```json


    "data_configs": [
        {
            "name": "dataset_default_train",
            "type": "HuggingfaceContainer",
            "params_config": {
                "data_name": "json", 
                "data_files": {
                    "type": "azureml_datastore",
                    "config": {
                        "azureml_client": {
                            "subscription_id": "396656ae-1e4b-4f9d-9a8a-a5fcb0296643",
                            "resource_group": "AIGroup",
                            "workspace_name": "kinfey-phi3-mini-demo-ws"
                        },
                        "datastore_name": "workspaceblobstore",
                        "relative_path": "UI/2024-05-20_030716_UTC/dataset-classification.json"
                    }
                },
                "split": "train",
                "component_kwargs": {
                    "pre_process_data": {
                        "dataset_type": "corpus",
                        "text_cols": [
                            "phrase",
                            "tone"
                        ],
                        "text_template": "### 文字: {phrase}\n### 語氣是:\n{tone}",
                        "corpus_strategy": "join",
                        "source_max_len": 1024
                    }
                }
            }
        }
    ],


```

**2. 計算設定**

如果你需要是本地的，你可以直接使用本地資料資源。你需要使用 Azure AI Studio / Azure Machine Learning Service 的資源。你需要設定相關的 Azure 參數、計算能力名稱等。

```json

    "systems": {
        "aml": {
            "type": "AzureML",
            "config": {
                "accelerators": ["gpu"],
                "hf_token": true,
                "aml_compute": "Your Azure AI Studio / Azure Machine Learning Service Compute Name",
                "aml_docker_config": {
                    "base_image": "Your Azure AI Studio / Azure Machine Learning Service docker",
                    "conda_file_path": "conda.yaml"
                }
            }
        },
        "azure_arc": {
            "type": "AzureML",
            "config": {
                "accelerators": ["gpu"],
                "aml_compute": "Your Azure AI Studio / Azure Machine Learning Service Compute Name",
                "aml_docker_config": {
                    "base_image": "Your Azure AI Studio / Azure Machine Learning Service docker",
                    "conda_file_path": "conda.yaml"
                }
            }
        }
    },

```

***注意***

因為它是通過 Azure AI Studio/Azure Machine Learning Service 上的容器執行的，所以需要配置所需的環境。這是在 conda.yaml 環境中配置的。

```yaml

name: project_environment
channels:
  - defaults
dependencies:
  - python=3.8.13
  - pip=22.3.1
  - pip:
      - einops
      - accelerate
      - azure-keyvault-secrets
      - azure-identity
      - bitsandbytes
      - datasets
      - huggingface_hub
      - peft
      - scipy
      - sentencepiece
      - torch>=2.2.0
      - transformers
      - git+https://github.com/microsoft/Olive@jiapli/mlflow_loading_fix#egg=olive-ai[gpu]
      - --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ 
      - ort-nightly-gpu==1.18.0.dev20240307004
      - --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
      - onnxruntime-genai-cuda

    
```

**3. 選擇你的 SLM**

你可以直接從 Hugging face 使用該模型，或者你可以直接將其與 Azure AI Studio / Azure Machine Learning 的模型目錄結合使用來選擇要使用的模型。這裡我以 Microsoft Phi-3-mini 為例。

如果你在本地有這個 model，你可以使用這個方法

```json

    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "hf_config": {
                "model_name": "model-cache/microsoft/phi-3-mini",
                "task": "text-generation",
                "model_loading_args": {
                    "trust_remote_code": true
                }
            }
        }
    },

```

如果你想使用來自 Azure AI Studio / Azure Machine Learning Service 的模型，你可以使用這個方法

```json

    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "model_path": {
                "type": "azureml_registry_model",
                "config": {
                    "name": "microsoft/Phi-3-mini-4k-instruct",
                    "registry_name": "azureml-msr",
                    "version": "11"
                }
            },
             "model_file_format": "PyTorch.MLflow",
             "hf_config": {
                "model_name": "microsoft/Phi-3-mini-4k-instruct",
                "task": "text-generation",
                "from_pretrained_args": {
                    "trust_remote_code": true
                }
            }
        }
    },

```

*注意:*

我們需要整合 Azure AI Studio / Azure Machine Learning Service，所以在設定模型時，請參考版本號和相關命名。

所有 Azure 上的模型需要設定為 PyTorch.MLflow

你需要擁有一個 Hugging face 帳戶並將密鑰綁定到 Azure AI Studio / Azure Machine Learning 的 Key 值

**4. 演算法**

Microsoft Olive 很好地封裝了 Lora 和 QLora 微調算法。你只需要設定一些相關參數。這裡我以 QLora 為例。

```json
        "lora": {
            "type": "LoRA",
            "config": {
                "target_modules": [
                    "o_proj",
                    "qkv_proj"
                ],
                "double_quant": true,
                "lora_r": 64,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "train_data_config": "dataset_default_train",
                "eval_dataset_size": 0.3,
                "training_args": {
                    "seed": 0,
                    "data_seed": 42,
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "gradient_checkpointing": false,
                    "learning_rate": 0.0001,
                    "num_train_epochs": 3,
                    "max_steps": 10,
                    "logging_steps": 10,
                    "evaluation_strategy": "steps",
                    "eval_steps": 187,
                    "group_by_length": true,
                    "adam_beta2": 0.999,
                    "max_grad_norm": 0.3
                }
            }
        },
```

如果你想要量化轉換，Microsoft Olive 主分支已經支援 onnxruntime-genai 方法。你可以根據你的需求設定：

1. 合併 adapter 權重到 base model
2. 使用 ModelBuilder 以所需精度將 model 轉換為 onnx model

例如轉換為量化的 INT4

```json

        "merge_adapter_weights": {
            "type": "合併適配器權重"
        },
        "builder": {
            "type": "模型建構器",
            "config": {
                "precision": "int4"
            }
        }

```

***注意***

- 如果你使用 QLoRA，目前暫不支援 ONNXRuntime-genai 的量化轉換。

- 這裡需要指出的是，你可以根據自己的需求來設定上述步驟。沒有必要完全配置上述這些步驟。根據你的需求，你可以直接使用演算法的步驟而不進行微調。最後你需要配置相關的引擎。

```json

    "engine": {
        "log_severity_level": 0,
        "host": "aml",
        "target": "aml",
        "search_strategy": false,
        "execution_providers": ["CUDAExecutionProvider"],
        "cache_dir": "../model-cache/models/phi3-finetuned/cache",
        "output_dir" : "../model-cache/models/phi3-finetuned"
    }

```

**5. 完成微調**

在命令列中，在 olive-config.json 的目錄中執行

```bash


olive run --config olive-config.json  


```

