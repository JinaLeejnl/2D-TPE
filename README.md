# 2D-TPE: Two-Dimensional Positional Encoding Enhances Table Understanding for Large Language Models

This repository is the official implementation of 2D-TPE: Two-Dimensional Positional Encoding Enhances Table Understanding for Large Language Models.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

You can find the training and testing data at [osunlp/TableInstruct](https://huggingface.co/datasets/osunlp/TableInstruct).

## Training

To train the model(s) in the paper, run this command:

```train
cd src
./run.sh
```

Replace `model_name_or_path`, `output_dir`, and `data_path` with the paths to your local model, the trained model's output directory, and the location of the training data, respectively.

## Evaluation

### Inference

```eval
cd src
python inference.py
```

### Evaluate

>evaluate various metrics
```eval
cd eval_scripts
python eval_hitab.py
```

## Pre-trained Models

You can download pretrained models here:

- [openbmb/MiniCPM-2B-sft-bf16-llama-format](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16-llama-format)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
