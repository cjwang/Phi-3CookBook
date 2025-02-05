﻿# **推論你的微調模型**

經過微調後，您可以透過引用存取新模型，這裡使用 ONNX Runtime GenAI 來實現。

## **安裝 ORT GenAI SDK**

**注意** - 請先安裝 CUDA 12.1，如果你不知道如何安裝，請閱讀此指南 [https://developer.nvidia.com/cuda-12-1-0-download-archive](https://developer.nvidia.com/cuda-12-1-0-download-archive)

after finish cuda, please install onnxruntime genai sdk with cuda

完成 CUDA 後，請安裝 ONNXRuntime GenAI SDK 並使用 CUDA

```bash

pip install numpy

pip install onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/

```

## **推論模型**

```python

import onnxruntime_genai as og

model = og.Model('Your onnx model folder location')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

search_options = {"max_length": 1024,"temperature":0.3}

params = og.GeneratorParams(model)
params.try_use_cuda_graph_with_max_batch_size(1)
params.set_search_options(**search_options)

prompt = "prompt = "<|user|>Who are you not allowed to marry in the UK?<|end|><|assistant|>""
input_tokens = tokenizer.encode(prompt)
params.input_ids = input_tokens

generator = og.Generator(model, params)

while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)


```

### **測試您的結果**

![result](../../imgs/06/e2e/result.png)

