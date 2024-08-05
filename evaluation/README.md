# LLM-based Evaluation

## Overview
The SGP-Bench uses regular expression to extract the answers from LLMs.

```bash
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
```
This assumes that the LLM exactly follows the instruction and produce answers that has the exact pattern specified in the question. In addition, we provide LLM-based evaluation, where we eliminate the cases, where the LLM produced the right answer but not in the specified format.

This section explains how to perform LLM-based evaluation by looking at the parsed response files.

## Prerequisites

Some additional python packages has to be installed before performing evaluation.
```bash
pip install opencv-python
pip install word2number
```


## Evaluation Script
The evaluation script loads the merged model, runs it on the evaluation dataset, and computes performance metrics.

### Running the Evaluation

1. **Parse the answer file (*.html)**:
   After running the evaluation, two result files are save in the `result` directory, with the model responses saved in the (*.html) file. Run
   ```bash
   python parse_html.py
   ```
   to extract the `prompt conversation` (spg-bench question without code), `sampled message` (model response) and `result` (gt result) and save the parsed information in a `*_query.json` file.

2. **Extract answer using a LLM**:
   Run the `result
   ```bash
   python extract_answer.py --result_file *_query.json
   ```

   ```bash
   python extract_answer.py --result_file $QUERY_FILE --llm_engine $LLM_ENGINE --base_url $BASE_URL

   # example
   python extract_answer.py --result_file *_query.json --llm_engine meta-llama/Meta-Llama-3.1-8B-Instruct --base_url 
   ```

### Example Script

Here's an example of how to run the evaluation:



## Acknowledgment
The code and methodology for this evaluation are adapted from the following GitHub project: [MathVista](https://github.com/lupantech/MathVista)