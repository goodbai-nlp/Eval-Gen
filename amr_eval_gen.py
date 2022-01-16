# coding:utf-8
import sys
import os
import argparse
from typing import Iterable, Optional
import datasets
import re


def argument_parser():

    parser = argparse.ArgumentParser(description="Preprocess AMR data")
    # Multiple input parameters
    parser.add_argument("--in-tokens", help="input tokens", required=True, type=str)
    parser.add_argument("--in-reference-tokens", help="refrence tokens to compute metric", type=str)
    parser.add_argument("--lower-case", help="use lower-case evaluation", default=False, action="store_true")
    args = parser.parse_args()

    return args


def tokenize_sentence(text, debug=False):
    text = re.sub(r"('ll|n't|'m|'s|'d|'re)", r" \1", text)
    text = re.sub(r"(\s+)", r" ", text)
    return text


def raw_corpus_bleu(
    hypothesis: Iterable[str], reference: Iterable[str], offset: Optional[float] = 0.01
) -> float:
    bleu = datasets.load_metric("bleu")
    hypothesis = [itm.strip().split() for itm in hypothesis]
    reference = [[itm.strip().split()] for itm in reference]
    res = bleu.compute(predictions=hypothesis, references=reference)
    return res


def raw_corpus_chrf(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    chrf = datasets.load_metric("chrf")
    hypotheses = [itm.strip() for itm in hypotheses]
    references = [[itm.strip()] for itm in references]
    res = chrf.compute(predictions=hypotheses, references=references)
    return res


def raw_corpus_meteor(hypotheses: Iterable[str], references: Iterable[str]):
    """Meteor v1.0, out-of-date, using java Meteor 1.5 instead!"""
    hypotheses = [itm.strip() for itm in hypotheses]
    references = [[itm.strip()] for itm in references]
    meteor = datasets.load_metric("meteor")
    res = meteor.compute(predictions=hypotheses, references=references)
    return res


def raw_corpus_bleurt(hypotheses: Iterable[str], references: Iterable[str]):
    hypotheses = [itm.strip() for itm in hypotheses]
    references = [itm.strip() for itm in references]
    bleurt = datasets.load_metric("bleurt", 'bleurt-base-512')
    res = bleurt.compute(predictions=hypotheses, references=references)
    return res


def raw_corpus_distinct(hypotheses: Iterable[str]):
    hypotheses = [itm.strip() for itm in hypotheses]
    distinct = datasets.load_metric('./distinct')
    res = distinct.compute(predictions=hypotheses)
    return res

def read_tokens(in_tokens_file):
    with open(in_tokens_file) as fid:
        lines = fid.readlines()
    return lines


if __name__ == "__main__":

    # Argument handlig
    args = argument_parser()

    # read files
    ref = read_tokens(args.in_reference_tokens)
    hyp = read_tokens(args.in_tokens)

    # Lower case output
    if args.lower_case:
        for i in range(len(ref)):
            ref[i] = ref[i].lower()
            hyp[i] = hyp[i].lower()

    
    # for i in range(len(hyp)):
    #     hyp[i] = tokenize_sentence(hyp[i])

    # results

    bleu = raw_corpus_bleu(hyp, ref)
    print("BLEU {}".format(bleu))

    chrFpp = raw_corpus_chrf(hyp, ref)
    print("chrF++ {}".format(chrFpp))

    #meteor = raw_corpus_meteor(hyp, ref)
    #print("meteor {}".format(meteor))

    #bleurt = raw_corpus_bleurt(hyp, ref)
    #b_res = sum(bleurt["scores"]) / len(bleurt["scores"])
    #print("bleurt {}".format(b_res))
