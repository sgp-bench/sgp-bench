"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re
import os

import pandas

from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from .custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult

from datasets import load_dataset



class SGPEval(Eval):
    def __init__(self, num_examples: int | None = None, mode: str = "raw", api: str = "default"):        
        dataset = load_dataset('sgp-bench/sgp-bench', split=mode)
        examples = [item for item in dataset]

        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.api = api

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            max_retries = 5
            retry_count = 0
            extracted_answer = None
            response_text = ""
            prompt_messages = [
                sampler._pack_message(content=format_multichoice_question(row), role="user")
            ]

            while retry_count < max_retries and extracted_answer is None:
                response_text = sampler(prompt_messages)
                match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
                extracted_answer = match.group(1) if match else None
                retry_count += 1

            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = row["Subject"]
            return SingleEvalResult(html=html, score=score, metrics={category: score}, convo=convo)


        if 'haiku' in self.api:
            results = common.map_with_progress(fn, self.examples, num_threads=2)
        elif 'sonnet' in self.api:
            results = common.map_with_progress(fn, self.examples, num_threads=5)
        elif 'openai' in self.api:
            results = common.map_with_progress(fn, self.examples, num_threads=5)
        else:
            results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
