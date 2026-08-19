---
language:
- en
license: mit
tags:
- Large Language Models
- Reasoning Models
- Mathematical Reasoning
- mathematics
task_categories:
- question-answering
features:
- name: question_id
  dtype: int64
- name: prompt
  dtype: string
- name: solution
  dtype: string
  description: "A detailed reasoning path written by human experts."
- name: answer
  dtype: string
- name: answer_type
  dtype: string
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: question_id
    dtype: int64
  - name: prompt
    dtype: string
  - name: solution
    dtype: string
    description: "A detailed reasoning path written by human experts."
  - name: answer
    dtype: string
  - name: answer_type
    dtype: string
  splits:
  - name: test
    num_bytes: 205982
    num_examples: 50
  download_size: 100150
  dataset_size: 205982
---

# 📐 AMO-Bench: Large Language Models Still Struggle in High School Math Competitions

- 📄 [Paper](https://huggingface.co/papers/2510.26768)
- 🌐 [Project Page](https://amo-bench.github.io/)
- 💻 [Github Repo](https://github.com/meituan-longcat/AMO-Bench)

## Updates
- 2026.02.05: Leaderboard Update: **[Qwen3-Max-Thinking](https://qwen.ai/blog?id=qwen3-max-thinking) achieves a new SOTA with 65.1%, while [GLM-4.7](https://z.ai/blog/glm-4.7) sets a new open-source record at 62.4%!** 
- 2025.12.01: We have added [Token Efficiency](#-token-efficiency) showing the number of output tokens used by models in the leaderboard. **[Gemini 3 Pro](https://deepmind.google/models/gemini/pro/) achieves the highest token efficiency among top-performance models!**
- 2025.11.24: **[Gemini 3 Pro](https://deepmind.google/models/gemini/pro/) achieves 63.1%, setting a new SOTA and breaking 60% for the first time!** We have updated the [Leaderboard](#-leaderboard) with the results of Gemini 3 Pro and [Qwen3-Max-Thinking (Preview)](https://qwen.ai/blog?id=qwen3-max).
- 2025.11.19: [Kimi-K2-Thinking](https://moonshotai.github.io/Kimi-K2/thinking.html) achieves 56.0%, new SOTA on [Leaderboard](#-leaderboard)!
- 2025.11.05: The problem statement of **Problem 35** has been revised: (1) the five integers that sum to $k$ should be **non-negative** rather than positive, and (2) we also stipulate that 1 couldn't be replaced with five integers. Additionally, for the strictly positive case in the original problem statement, the correct answer should be 7656 (see this [discussion](https://huggingface.co/datasets/meituan-longcat/AMO-Bench/discussions/4) for details). Thanks to the feedback from [@applesilicon](https://huggingface.co/datasets/meituan-longcat/AMO-Bench/discussions/4)!
- 2025.10.31: We release the dataset, evaluation code, and technical report of AMO-Bench.

## 📊 Leaderboard

<p align="center">
    <img src="https://github.com/meituan-longcat/AMO-Bench/raw/main/figures/leaderboard_20260205.png" width="800">
</p>

## 📈 Token Efficiency

<p align="center">
    <img src="https://github.com/meituan-longcat/AMO-Bench/raw/main/figures/acc_vs_len_20260205.png" width="800">
</p>

## 📝 Abstract

We present AMO-Bench, an Advanced Mathematical reasoning benchmark with Olympiad level or even higher difficulty, comprising 50 human-crafted problems. Existing benchmarks have widely leveraged high school math competitions for evaluating mathematical reasoning capabilities of large language models (LLMs). However, many existing math competitions are becoming less effective for assessing top-tier LLMs due to performance saturation (e.g., AIME24/25). To address this, AMO-Bench introduces more rigorous challenges by ensuring all 50 problems are (1) cross-validated by experts to meet at least the International Mathematical Olympiad (IMO) difficulty standards, and (2) entirely original problems to prevent potential performance leakages from data memorization. Moreover, each problem in AMO-Bench requires only a final answer rather than a proof, enabling automatic and robust grading for evaluation. Experimental results across 26 LLMs on AMO-Bench show that even the best-performing model achieves only 52.4% accuracy on AMO-Bench, with most LLMs scoring below 40%. Beyond these poor performances, our further analysis reveals a promising scaling trend with increasing test-time compute on AMO-Bench. These results highlight the significant room for improving the mathematical reasoning in current LLMs. We release AMO-Bench to facilitate further research into advancing the reasoning abilities of language models.

<p align="center">
    <img src="https://github.com/meituan-longcat/AMO-Bench/raw/main/figures/cover_fig.png" width="800">
</p>

## ⭐ Key Features

<p align="center">
    <img src="https://github.com/meituan-longcat/AMO-Bench/raw/main/figures/pipeline.png" width="800">
</p>

- **Original problems.** To prevent performance leaks from existing resources as much as possible,
all problems in AMO-Bench are newly crafted by human experts. Moreover, we conduct a
secondary verification to ensure that there are no highly similar problems in existing competitions
or online resources.
- **Guaranteed difficulty.** Each problem has undergone rigorous cross-validation by multiple
experts to ensure it meets at least the difficulty standards of IMO. We also incorporate an
LLM-based difficulty filtering stage to exclude questions that do not present sufficient challenge
to current reasoning models.
- **Final-answer based grading.** Each problem in AMO-Bench requires a final answer rather than
a full proof, enabling efficient automatic grading. For each problem, we employ a parser-based
or LLM-based grading method according to its answer type, balancing the grading cost and
generalizability.
- **Human-annotated reasoning paths.** In addition to the final answer, each problem also includes
a detailed reasoning path written by human experts. These additional annotations enhance
solution transparency and could support further explorations on AMO-Bench, such as prompt
engineering and error analysis.


## 📖 Dataset Description

**AMO-Bench** is a curated test set designed to benchmark advanced mathematical reasoning. Its creation was guided by the following core features to ensure a fair and challenging evaluation:

- **Original Problems**: To prevent performance leaks from existing resources as much as possible, all problems in AMO-Bench are newly crafted by human experts. Moreover, we conduct a secondary verification to ensure that there are no highly similar problems in existing competitions or online resources.

- **Guaranteed Difficulty**: Each problem has undergone rigorous cross-validation by multiple experts to ensure it meets at least the difficulty standards of IMO. We also incorporate an LLM-based difficulty filtering stage to exclude questions that do not present sufficient challenge to current reasoning models.

- **Final-Answer Based Grading**, Each problem in AMO-Bench requires a final answer rather than a full proof, enabling efficient automatic grading. For each problem, we employ a parser-based or LLM-based grading method according to its answer type, balancing the grading cost and generalizability.

- **Human-Annotated Reasoning Paths**: In addition to the final answer, each problem also includes a detailed reasoning path written by human experts. These additional annotations enhance solution transparency and could support further explorations on AMO-Bench, such as prompt engineering and error analysis.


## 📊 Data Fields

Each entry in the dataset consists of five fields:

- `question_id`: (`int64`) - The ID of the mathematical problem.

- `prompt`: (`string`) - A full statement of the mathematical problem, formatted in Markdown with LaTeX support for mathematical expressions.

- `solution`: (`string`) - A detailed reasoning path written by human experts.

- `answer`: (`string`) - The final answer to the problem.

- `answer_type`: (`string`) - The types of the final answer, including four main types.

Here is an example of a data instance:
```json
{
  "question_id": 4, 
  "prompt": "Let $x_1,x_2,\\ldots,x_{2025}$ be real numbers in the interval $[0,1]$, and let $f(x)$ be a real-valued function ...",
  "solution": "Let\n$$\nm=\\max_{x_1,x_2,\\dots,x_{2025}}\\left|\\sum_{k=1}^{2025}f(x_k)-\\left(\\sum_{k=1}^{2025}x_k\\right)^2\\right|.\n$$\nWe have ...", 
  "answer": "\\boxed{512578}", 
  "answer_type": "number"
  }
```

## ✍️ Dataset Construction

**AMO-Bench** have built up a comprehensive multi-stage construction pipeline that covers the entire process from question creation to final inclusion. This pipeline comprises four major stages: 

**(1) Data creation.** All problems are independently designed by mathematics experts from top universities and educational institutions. Beyond the final answer, each problem author must provide a detailed step-by-step solution. 

**(2) Quality review.** Each candidate problem undergoes blind review by at least three experts to assess its quality. 

**(3) Originality review.** The originality review stage aims to ensure that these newly created problems are not mere rewrites of publicly available materials, but demonstrate genuine originality. 

**(4) Difficulty review.** We implement a difficulty review stage to filter out problems lacking adequate complexity, to ensure that AMO-Bench presents a sufficient challenge to state-of-the-art LLMs.

## 🚀 Sample Usage (and Evaluation)

This section outlines how to quickly get started with the AMO-Bench dataset and evaluate model performance. For more detailed guidelines, please refer to the [GitHub repository](https://github.com/meituan-longcat/AMO-Bench).

### Installation

1. Clone the repository:
```bash
git clone https://github.com/meituan-longcat/AMO-Bench.git
cd AMO-Bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running evaluations

#### Step 1: Format Model Response File

After obtaining model responses, format them as follows (one JSON object per line):
```data
{"question_id": 1, "model_response": "..."}
{"question_id": 2, "model_response": "..."}
...
```
Save this file in the `./model_responses/` directory.

#### Step 2: Grading Responses
Set your API key and URL in lines 13-14 of `utils.py`.
Then run:
```bash
python grading.py --response_file example.jsonl
```
Evaluation results will be saved under the `./grading_results/` directory.

#### Step 3 (Optional): Grade on AMO-Bench-P Subset

For a quick evaluation using only the parser-based subset (39 problems), run:
```bash
python grading.py --response_file example.jsonl --only_parser True
```

## Discussions and Feedbacks
Here we summarize the discussions and feedbacks on AMO-Bench from the open-source community.
We will regularly update the dataset to address urgent data issues.

We welcome any feedback you may have!
- Problem 26 appears to be effectively the same as an existing contest problem. Thanks to [@applesilicon](https://huggingface.co/datasets/meituan-longcat/AMO-Bench/discussions/3) to point this out!
- The problem statement for Problem 35 should be further clarified: (1) the five integers that sum to $k$ should be **non-negative** rather than positive, and (2) we also stipulate that 1 couldn't be replaced with five integers. Additionally, for the strictly positive case in the original problem statement, the correct answer should be 7656 (see this [discussion](https://huggingface.co/datasets/meituan-longcat/AMO-Bench/discussions/4) for details). Thanks to the suggestions from [@applesilicon](https://huggingface.co/datasets/meituan-longcat/AMO-Bench/discussions/4)!
- Four problems involve complex numerical expressions (Problem 12, 13, 15 and 21). When tackling these problems, LLMs may struggle to perform accurate calculations without calling external tools. Thanks to the feedback from [@prnake](https://github.com/meituan-longcat/AMO-Bench/issues/1)!
- Problem 38 & 39 appear to be similar in content to two arXiv papers [[1]](https://arxiv.org/pdf/2508.19413) [[2]](https://arxiv.org/pdf/2508.03927).

## 📎 Citation

If you use the AMO-Bench dataset in your research or work, please consider citing it:

```bibtex
@misc{an2025amobench,
      title={AMO-Bench: Large Language Models Still Struggle in High School Math Competitions}, 
      author={Shengnan An and Xunliang Cai and Xuezhi Cao and Xiaoyu Li and Yehao Lin and Junlin Liu and Xinxuan Lv and Dan Ma and Xuanlin Wang and Ziwen Wang and Shuang Zhou},
      year={2025},
      eprint={2510.26768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.26768}, 
}
```

## ✅ License

AMO-Bench is released under the [MIT LICENSE](LICENSE).

## 🤝 Acknowledgement

The evaluation script utilizes [Math-Verify](https://github.com/huggingface/Math-Verify) to parse and verify model outputs.
We greatly appreciate the contributors' efforts in providing this valuable tool.

## 📩 Support

For questions and support, please open an issue on GitHub or contact the maintainers.