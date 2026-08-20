---
dataset_info:
  features:
  - name: source
    dtype: string
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: system
    dtype: string
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: generated_token_count
    dtype: int64
  - name: correct
    dtype: bool
  - name: Question
    dtype: string
  - name: COT_Reason
    dtype: string
  - name: Answer
    dtype: string
  splits:
  - name: train
    num_bytes: 653862134
    num_examples: 29434
  download_size: 279908279
  dataset_size: 653862134
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
