# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
this python script is used for compute the distinct1/2

follow this paper for more details:
https://aclanthology.org/N16-1014.pdf

this code borrows from:
https://github.com/golsun/NLP-tools/blob/master/metrics.py
'''
from collections import defaultdict

import datasets


_CITATION = """\
@InProceedings{huggingface:metric,
title = {distinct},
authors={ZiYang Huang},
year={2022}
}
"""


_DESCRIPTION = '''

distinct1/2 measures the generation diversity, which 
is defined as the number of distinct uni- or bi-grams 
divided by the total amount of generated words

follow this paper for more details:
https://aclanthology.org/N16-1014.pdf

this code borrows from:
https://github.com/golsun/NLP-tools/blob/master/metrics.py
'''


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    distinct1,
    distinct2
Examples:
    >>> distinct = datasets.load_metric("distinct")
    >>> results = distinct.compute(predictions=['i love love you'])
    >>> print(results)
    {'distinct1': 0.75, 'distinct2': 1}
"""

# # TODO: Define external resources urls if needed
# BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Distinct(datasets.Metric):
    """
    distinct1/2 measures the generation diversity, which
        is defined as the number of distinct uni- or bi-grams
        divided by the total amount of generated words
    """

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )
    #
    # def _download_and_prepare(self, dl_manager):
    #     """Optional: download external resources useful to compute the scores"""
    #     # TODO: Download external resources if needed
    #     bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
    #     self.bad_words = set([w.strip() for w in open(bad_words_path, "r", encoding="utf-8")])

    def _compute(self, predictions, references):
        """Returns the scores"""
        # TODO: Compute the different scores of the metric
        tokens = [0.0, 0.0]
        types = [defaultdict(int), defaultdict(int)]
        for line in predictions:
            words = line.strip('\n').split()
            for n in range(2):
                for idx in range(len(words) - n):
                    ngram = ' '.join(words[idx:idx + n + 1])
                    types[n][ngram] = 1
                    tokens[n] += 1
        div1 = 1. * len(types[0].keys()) / tokens[0]
        div2 = 1. * len(types[1].keys()) / tokens[1]

        return {
            "distinct1": div1,
            "distinct2": div2,
        }