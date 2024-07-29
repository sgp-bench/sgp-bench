"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import random
import re
import os

import blobfile as bf
import pandas

from . import common
from .common import ANSWER_PATTERN, HTML_JINJA, check_equality
from .custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult

QUERY_TEMPLATE = """
Solve the following problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Important, put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


class MNISTEval(Eval):
    def __init__(self, equality_checker: SamplerBase, num_examples: int | None = None, mode: str = "mnist", api: str = "default"):
        csv_path = os.path.join(os.path.dirname(__file__), "data", f"sgp_{mode}_testset.csv")
        df = pandas.read_csv(
            # bf.BlobFile("https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv")
            csv_path
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.equality_checker = equality_checker
        self.api = api

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            max_retries = 5
            retry_count = 0
            extracted_answer = None
            response_text = ""
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")
            ]
            
            while retry_count < max_retries and extracted_answer is None:
                response_text = sampler(prompt_messages)
                match = re.search(ANSWER_PATTERN, response_text)
                extracted_answer = match.group(1) if match else None
                retry_count += 1

            score = float(check_equality(self.equality_checker, row["Answer"], extracted_answer))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)


        if 'haiku' in self.api:
            results = common.map_with_progress(fn, self.examples, num_threads=2)
        elif 'sonnet' in self.api:
            results = common.map_with_progress(fn, self.examples, num_threads=5)
        else:
            results = common.map_with_progress(fn, self.examples)

        results = common.map_with_progress(fn, self.examples, 50)
        return common.aggregate_results(results)
