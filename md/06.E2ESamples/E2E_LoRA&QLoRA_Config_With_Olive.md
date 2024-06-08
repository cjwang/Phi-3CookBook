# **使用 Microsoft Olive 來設計你的專案**

如果一個企業想要擁有自己的行業垂直模型，它需要從數據、微調和部署開始。在之前的內容中，我們介紹了 Microsoft Olive 的內容，現在我們基於 E2E 的工作完成更詳細的介紹。

## **架構**

我們可以參考由 AI Toolkit for VS Code 產生的專案來構建我們的專案，包括資料、模型、微調格式和推論。例如

```txt

｜-- 您的 Phi-3-mini E2E 專案
    ｜-- 資料集
    ｜-- 微調
    ｜-- 推論
    ｜-- 模型快取
    ｜-- 生成模型
    ｜-- 設定
     

```

- **datasets**

    資料可以儲存在 csv、json 和其他資料格式中。在此範例中，是匯出的 json 資料。[dataset](./E2E_Datasets.md) 

    ***注意*** 我們可以忽略這裡的相關設定，因為資料已經上傳到 Azure ML（如果是本地資料，我們可以在這裡上傳）

- **fine-tuning**
    
    指定微調 QLoRA 和 LoRA 演算法，以及相關參數

- **inferences**

    推論是微調後的模型。它可以是微調後的 Adapter 層參考，微調後與 Adapter 整合的模型參考，或者是量化的 ONNX Runtime 模型。

- **model-cache**

    通過 Hugging face CLI 下載的模型，這裡是 Phi-3-Mini 模型（使用 Azure ML 我們可以忽略這部分內容，如果你想在本地操作，請執行以下腳本以獲取 phi-3 模型）

```bash

huggingface-cli login

# 輸入你的 Hugging Face Portal 金鑰

huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir Your Phi-3-mini location

```

- **gen-model**

模型在操作後儲存，包括微調的 Adapter 模型、整合的微調 Adapter 模型，以及由 ONNX Runtime 執行的量化模型。

- **設定**

必須的安裝環境，請執行此操作來設定你的 Olive 環境

```bash

pip install -r requirements.txt

```

## **Microsoft Olive Config**

如果你想了解 Microsoft Olive 的設定，請訪問 [使用 Microsoft Olive 進行微調](../04.Fine-tuning/FineTuning_MicrosotOlive.md)

***注意*** 若要保持最新狀態，請使用以下指令安裝 Microsoft Olive

```bash

pip install git+https://github.com/microsoft/Olive

```

## **在 Azure ML 執行 Microsoft Olive**

**LoRA**

這個範例使用雲端運算，雲端資料集，添加 olive.config 在微調資料夾中

```json

{
    "azureml_client": {
        "subscription_id": "Your Azure Subscription ID",
        "resource_group": "Your Azure Resource Group",
        "workspace_name": "Your Azure ML Worksapce",
        "keyvault_name":  "Your Azure Key Valuts"
    },
    "input_model":{
        "type": "PyTorchModel",
        "config": {
            "hf_config": {
                "model_name": "microsoft/Phi-3-mini-4k-instruct",
                "task": "text-generation",
                "from_pretrained_args": {
                    "trust_remote_code": true
                }
            }
        }
    },
    "systems": {
        "aml": {
            "type": "AzureML",
            "config": {
                "accelerators": [
                    {
                        "device": "gpu",
                        "execution_providers": [
                            "CUDAExecutionProvider"
                        ]
                    }
                ],
                "hf_token": true,
                "aml_compute": "Your Azure ML Compute Cluster",
                "aml_docker_config": {
                    "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
                    "conda_file_path": "conda.yaml"
                }
            }
        },
        "azure_arc": {
            "type": "AzureML",
            "config": {
                "accelerators": [
                    {
                        "device": "gpu",
                        "execution_providers": [
                            "CUDAExecutionProvider"
                        ]
                    }
                ],
                "aml_compute": "Your Azure ML Compute",
                "aml_docker_config": {
                    "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
                    "conda_file_path": "conda.yaml"
                }
            }
        }
    },
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
                            "subscription_id": "Your Azure Subscrition ID",
                            "resource_group": "Your Azure Resource Group",
                            "workspace_name": "Your Azure ML Workspaces name"
                        },
                        "datastore_name": "workspaceblobstore",
                        "relative_path": "Your zhihu_train_data.json Azure ML Location"
                    }
                },
                "split": "train",
                "component_kwargs": {
                    "pre_process_data": {
                        "dataset_type": "corpus",
                        "text_cols": [
                            "INSTRUCTION",
                            "RESPONSE",
                            "SOURCE"
                        ],
                        "text_template": "<|user|>\n{INSTRUCTION}<|end|>\n<|assistant|>\n{RESPONSE}\n( source : {SOURCE})<|end|>",
                        "corpus_strategy": "join",
                        "source_max_len": 2048,
                        "pad_to_max_len": false,
                        "use_attention_mask": false
                    }
                }
            }
        }
    ],
    "passes": {
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
                "eval_dataset_size": 0.1,
                "training_args": {
                    "seed": 0,
                    "data_seed": 42,
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "gradient_checkpointing": false,
                    "learning_rate": 0.0001,
                    "num_train_epochs": 1000,
                    "max_steps": 100,
                    "logging_steps": 100,
                    "evaluation_strategy": "steps",
                    "eval_steps": 187,
                    "group_by_length": true,
                    "adam_beta2": 0.999,
                    "max_grad_norm": 0.3
                }
            }
        },
        "merge_adapter_weights": {
            "type": "MergeAdapterWeights"
        },
        "builder": {
            "type": "ModelBuilder",
            "config": {
                "precision": "int4"
            }
        }
    },
    "engine": {
        "log_severity_level": 0,
        "host": "aml",
        "target": "aml",
        "search_strategy": false,
        "cache_dir": "cache",
        "output_dir" : "../model-cache/models/phi3-finetuned"
    }
}

```

**QLoRA**

```json

{
    "azureml_client": {
        "subscription_id": "Your Azure Subscription ID",
        "resource_group": "Your Azure Resource Group",
        "workspace_name": "Your Azure ML Worksapce",
        "keyvault_name":  "Your Azure Key Valuts"
    },
    "input_model": {
        "type": "PyTorchModel",
        "config": {
            "hf_config": {
                "model_name": "microsoft/Phi-3-mini-4k-instruct",
                "task": "text-generation",
                "from_pretrained_args": {
                    "trust_remote_code": true
                }
            }
        }
    },
    "systems": {
        "aml": {
            "type": "AzureML",
            "config": {
                "accelerators": [
                    {
                        "device": "gpu",
                        "execution_providers": [
                            "CUDAExecutionProvider"
                        ]
                    }
                ],
                "hf_token": true,
                "aml_compute": "Your Azure ML Compute Cluster",
                "aml_docker_config": {
                    "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
                    "conda_file_path": "conda.yaml"
                }
            }
        },
        "azure_arc": {
            "type": "AzureML",
            "config": {
                "accelerators": [
                    {
                        "device": "gpu",
                        "execution_providers": [
                            "CUDAExecutionProvider"
                        ]
                    }
                ],
                "aml_compute": "Your Azure ML Compute",
                "aml_docker_config": {
                    "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
                    "conda_file_path": "conda.yaml"
                }
            }
        }
    },
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
                            "subscription_id": "Your Azure Subscrition ID",
                            "resource_group": "Your Azure Resource Group",
                            "workspace_name": "Your Azure ML Workspaces name"
                        },
                        "datastore_name": "workspaceblobstore",
                        "relative_path": "Your zhihu_train_data.json Azure ML Location"
                    }
                },
                "split": "train",
                "component_kwargs": {
                    "pre_process_data": {
                        "dataset_type": "corpus",
                        "text_cols": [
                            "INSTRUCTION",
                            "RESPONSE",
                            "SOURCE"
                        ],
                        "text_template": "<|user|>\n{INSTRUCTION}<|end|>\n<|assistant|>\n{RESPONSE}\n( source : {SOURCE})<|end|>",
                        "corpus_strategy": "join",
                        "source_max_len": 2048,
                        "pad_to_max_len": false,
                        "use_attention_mask": false
                    }
                }
            }
        }
    ],
    "passes": {
        "qlora": {
            "type": "QLoRA",
            "config": {
                "compute_dtype": "bfloat16",
                "quant_type": "nf4",
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
        "merge_adapter_weights": {
            "type": "MergeAdapterWeights"
        }
    },
    "engine": {
        "log_severity_level": 0,
        "host": "aml",
        "target": "aml",
        "search_strategy": false,
        "cache_dir": "cache",
        "output_dir": "../model-cache/models/phi3-finetuned"
    }
}

```

***注意***

- 如果你使用 QLoRA，目前不支援 ONNXRuntime-genai 的量化轉換。

- 這裡需要指出的是，你可以根據自己的需求設定上述步驟。沒有必要完全配置上述這些步驟。根據你的需求，你可以直接使用演算法的步驟而不進行微調。最後你需要配置相關的引擎。

### **執行 Microsoft Olive**

完成 Microsoft Olive 後，你需要在終端機中執行此命令

```bash

olive 執行 --config olive-config.json  

```

***注意***

1. 當 Microsoft Olive 被執行時，每個步驟都可以放置在快取中。我們可以通過查看微調目錄來查看相關步驟的結果。

![快取](../../imgs/06/e2e/cache.png)

2. 我們在這裡提供 LoRA 和 QLoRA ，您可以根據需要進行設定。

3. 建議的執行環境是 WSL / Ubuntu 22.04+。

4. 為什麼選擇 ORT？因為 ORT 可以部署在邊緣設備上，推論是在 ORT 環境中實現的。

![ort](../../imgs/06/e2e/ort.png)

