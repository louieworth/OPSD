---
dataset_info:
- config_name: default
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: answer
    dtype: string
  - name: solution
    dtype: string
  - name: url
    dtype: string
  - name: year
    dtype: int64
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 17407
    num_examples: 30
  download_size: 0
  dataset_size: 17407
- config_name: part1
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: answer
    dtype: string
  - name: solution
    dtype: string
  - name: url
    dtype: string
  - name: year
    dtype: int64
  splits:
  - name: train
    num_bytes: 6940
    num_examples: 15
  download_size: 9496
  dataset_size: 6940
- config_name: part2
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: answer
    dtype: int64
  - name: solution
    dtype: int64
  - name: url
    dtype: string
  - name: year
    dtype: int64
  splits:
  - name: train
    num_bytes: 10263
    num_examples: 15
  download_size: 15085
  dataset_size: 10263
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
- config_name: part1
  data_files:
  - split: train
    path: part1/train-*
- config_name: part2
  data_files:
  - split: train
    path: part2/train-*
---

# AIME 2025

This dataset contains 30 problems from the 2025 AIME tests, including:
- [**AIME I**](https://artofproblemsolving.com/wiki/index.php/2025_AIME_I): 15 problems
- [**AIME II**](https://artofproblemsolving.com/wiki/index.php/2025_AIME_II): 15 problems