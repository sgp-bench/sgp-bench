import os
import re
import time
import argparse

from tqdm import tqdm
import numpy as np
from utilities import get_chat_response, get_chat_response_gpt, read_json

# load demo prompt
from prompts.ext_ans import demo_prompt


ANSWER_PATTERN_MULTICHOICE = r"(?i)Extracted answer\s*:\s*([A-D])"


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Question: {query}\n\nModel response: {response}"
    full_prompt = f"{demo_prompt}\n\nNow, please look at the following Query-Anwser pair and extract the answer from the model response.\n\nHint: The last line of your response should be of the following format: 'Extracted answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n\n{test_prompt}\n\n"
    return full_prompt


def extract_answer(response, query, llm_engine, base_url, api_key):
    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        if 'gpt' in llm_engine:
            import openai
            os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            openai.api_key = os.getenv("OPENAI_API_KEY")
            extraction = get_chat_response_gpt(full_prompt, llm_engine, openai.api_key)
        else:
            extraction = get_chat_response(full_prompt, llm_engine, base_url, api_key)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--result_file', type=str, required=True, help='result file')
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--output_file', type=str, default='answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    # model
    parser.add_argument('--llm_engine', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='llm engine')
    parser.add_argument('--base_url', type=str, default='http://localhost:8000/v1', help='llm temperature')
    parser.add_argument('--api_key', type=str, default='token-abc123', help='open-sourced llm api key')
    parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    parser.add_argument('--output_label', type=str, default='', help='label for the output file')
    args = parser.parse_args()

    output_file = args.result_file.replace('.json', f'_LLM_extracted.txt')

    # read results
    print(f"Reading {args.result_file}...")
    results_list = read_json(os.path.join(args.output_dir, args.result_file))
    results = np.array(results_list).reshape(-1, 3)

    all_acc = []
    for pid in tqdm(range(len(results))):
    # for pid in tqdm(range(3)):
        problem = results[pid]
        # import ipdb; ipdb.set_trace()

        query = problem[0]["prompt conversation"]   
        response = problem[1]["sampled message"]    
        gt = problem[2]["results"].split('Correct Answer: ')[1].strip()

        max_retries = 5
        retry_count = 0
        extraction = None
        while retry_count < max_retries and extraction is None:
            response_text = extract_answer(response, query, args.llm_engine, args.base_url, args.api_key)
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extraction = match.group(1) if match else None
            retry_count += 1

        results[pid][2]['extraction'] = extraction
        all_acc.append(extraction == gt)
        if extraction == gt:
            results[pid][2]['correct'] = 1
        else:
            results[pid][2]['correct'] = 0

    avg_results = np.array(all_acc).mean()
    print(f"Average accuracy: {avg_results}")
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as file:
        file.write(str(avg_results))
    
