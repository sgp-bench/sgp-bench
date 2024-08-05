"""
This file contains code adapted from the following project:
    https://github.com/openai/simple-evals

Original project authors: OpenAI

Original project license: MIT License
"""

import os
import json
import time
import argparse

import pandas as pd

from . import common
from .sgp_eval import SGPEval
from .mnist_eval import MNISTEval
from .sampler.gpt_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OpenaiChatCompletionSampler,
)

from .sampler.claude_sampler import (
    ClaudeCompletionSampler, 
    CLAUDE_SYSTEM_MESSAGE_API,
    CLAUDE_SYSTEM_MESSAGE_LMSYS,
)

from .sampler.open_sampler import (
    OpenChatCompletionSampler,
    OPEN_SYSTEM_MESSAGE_API,
)

os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["ANTHROPIC_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["OPEN_API_KEY"] = "token-abc123"


def task_parser(task_list):
    parsed_list = []

    if 'svg' in task_list:
        parsed_list.append('sgp_svg')

    if 'cad' in task_list:
        parsed_list.append('sgp_cad')

    if 'mnist' in task_list:
        parsed_list.append('sgp_mnist')

    if 'inv' in task_list:
        parsed_list.extend([
            "sgp_inv", "sgp_inv_t0", "sgp_inv_t1", "sgp_inv_t2", "sgp_inv_t3", "sgp_inv_t4", 
            "sgp_inv_r0", "sgp_inv_r1", "sgp_inv_r2", "sgp_inv_r3", "sgp_inv_r4"
        ])

    return parsed_list



def main(args):
    debug = args.debug
    if args.api == "openai-4o":
        samplers = {
            "gpt-4o-2024-05-13": OpenaiChatCompletionSampler(
                model="gpt-4o-2024-05-13",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
            )
        }
    elif args.api == "openai-4o-mini":
        samplers = {
            "gpt-4o-mini-2024-07-18": OpenaiChatCompletionSampler(
                model="gpt-4o-mini-2024-07-18",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
            )
        }
    elif args.api == "openai-4":
        samplers = {
            "gpt-4-turbo-2024-04-09": OpenaiChatCompletionSampler(
                model="gpt-4-turbo-2024-04-09",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
            )
        }
    elif args.api == "openai-3.5":
        samplers = {
            "gpt-3.5-turbo-0125": OpenaiChatCompletionSampler(
                model="gpt-3.5-turbo-0125",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
            )
        }
    elif args.api == "claude-3.5-sonnet":
        samplers = {
            "claude-3-5-sonnet-20240620": ClaudeCompletionSampler(
                model="claude-3-5-sonnet-20240620", 
                system_message=CLAUDE_SYSTEM_MESSAGE_API,
            ),
        }
    elif args.api == "claude-3-opus":
        samplers = {
            "claude-3-opus-20240229": ClaudeCompletionSampler(
                model="claude-3-opus-20240229", 
                system_message=CLAUDE_SYSTEM_MESSAGE_API,
            ),
        }
    elif args.api == "claude-3-sonnet":
        samplers = {
            "claude-3-sonnet-20240229": ClaudeCompletionSampler(
                model="claude-3-sonnet-20240229", 
                system_message=CLAUDE_SYSTEM_MESSAGE_API,
            ),
        }
    elif args.api == "claude-3-haiku":
        samplers = {
            "claude-3-haiku-20240307": ClaudeCompletionSampler(
                model="claude-3-haiku-20240307", 
                system_message=CLAUDE_SYSTEM_MESSAGE_API,
            ),
        }
    else:
        if "gemma" in args.api or "mistral" in args.api or "starcoder" in args.api:
            samplers = {
                args.api: OpenChatCompletionSampler(
                    model=args.model,
                    system_message=None,
                    base_url=args.base_url
                ),
            }
        elif "aya" in args.api:
            samplers = {
                args.api: OpenChatCompletionSampler(
                    model=args.model,
                    system_message=OPEN_SYSTEM_MESSAGE_API,
                    max_tokens=args.max_tokens,
                    base_url=args.base_url
                ),
            }
        else:
            samplers = {
                args.api: OpenChatCompletionSampler(
                    model=args.model,
                    system_message=OPEN_SYSTEM_MESSAGE_API,
                    base_url=args.base_url
                ),
            }

    equality_checker = OpenaiChatCompletionSampler(model="gpt-4o")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, api):
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "sgp_svg":
                return SGPEval(num_examples=10 if debug else None, mode="svg", api=api)
            case "sgp_cad":
                return SGPEval(num_examples=10 if debug else None, mode="cad", api=api)
            case "sgp_mnist":
                return MNISTEval(equality_checker=equality_checker, num_examples=10 if debug else None, mode="mnist", api=api)
            case "sgp_inv":
                return SGPEval(num_examples=10 if debug else None, mode="inv", api=api)
            case "sgp_inv_t0":
                return SGPEval(num_examples=10 if debug else None, mode="inv_t0", api=api)
            case "sgp_inv_t1":
                return SGPEval(num_examples=10 if debug else None, mode="inv_t1", api=api)
            case "sgp_inv_t2":
                return SGPEval(num_examples=10 if debug else None, mode="inv_t2", api=api)
            case "sgp_inv_t3":
                return SGPEval(num_examples=10 if debug else None, mode="inv_t3", api=api)
            case "sgp_inv_t4":
                return SGPEval(num_examples=10 if debug else None, mode="inv_t4", api=api)
            case "sgp_inv_r0":
                return SGPEval(num_examples=10 if debug else None, mode="inv_r0", api=api)
            case "sgp_inv_r1":
                return SGPEval(num_examples=10 if debug else None, mode="inv_r1", api=api)
            case "sgp_inv_r2":
                return SGPEval(num_examples=10 if debug else None, mode="inv_r2", api=api)
            case "sgp_inv_r3":
                return SGPEval(num_examples=10 if debug else None, mode="inv_r3", api=api)
            case "sgp_inv_r4":
                return SGPEval(num_examples=10 if debug else None, mode="inv_r4", api=api)
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")

    eval_tasks = task_parser(args.eval)
    evals = {
        eval_name: get_evals(eval_name, args.api) for eval_name in eval_tasks
    }

    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    mergekey2resultpath = {}
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)
    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = os.path.join(result_dir, f"{file_stem}{debug_suffix}.html")
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}

            # Print metrics 
            filtered_data = {key: value for key, value in metrics.items() if not key.endswith(':std')}
            df = pd.DataFrame(filtered_data, index=[sampler_name]).reset_index()
            columns = ['sampler_name'] + [('metric', key) for key in filtered_data.keys()]
            df.columns = columns
            df.set_index('sampler_name', inplace=True)
            print("\nResults: ")
            print(df.to_markdown())

            result_filename = os.path.join(result_dir, f"{file_stem}{debug_suffix}.json")
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_sampler_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_sampler_name[: eval_sampler_name.find("_")]
        sampler_name = eval_sampler_name[eval_sampler_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    return merge_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://172.22.8.7:8000/v1")
    parser.add_argument("--api", type=str, default="llama3.1-8B")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval", type=str, nargs='+', default=["svg"], help="List of evaluation tasks", choices=["svg", "cad", "mnist", "inv"])
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    main(args)
