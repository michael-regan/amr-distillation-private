#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes sembleu score between two AMRs represented as ngrams
For detailed description of sembleu, see https://github.com/freesunshine0316/sembleu/

"""
import sys
sys.path.insert(0, "/home/michaelr/packages")
import sembleu

import amr_parser

from sembleu import src
from sembleu.src import bleu_score
# from sembleu.src.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction, NgramInst
from sembleu.src.bleu_score import SmoothingFunction, NgramInst
from sembleu.src import amr_graph
from sembleu.src.amr_graph import AMRGraph

import fractions
try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

import math
from collections import Counter


def brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1.0
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0.0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

def closest_ref_length_amr(references, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: The length of the hypothesis.
    :type hypothesis: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    ref_lens = (reference.length for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
                          (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len

def modified_precision_amr(references, hypothesis, n):
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(hypothesis.ngram[n]) if n in hypothesis.ngram else Counter()
    #print 'counts', counts
    # Extract a union of references' counts.
    ## max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = Counter(reference.ngram[n]) if n in reference.ngram else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])
    #print('max_counts', max_counts)

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {ngram: min(count, max_counts[ngram])
                      for ngram, count in list(counts.items())}
    #print('clipped_counts', clipped_counts)

    numerator = sum(clipped_counts.values())
    denominator = sum(counts.values())
    ## Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    ## Usually this happens when the ngram order is > len(reference).
    #denominator = max(1, sum(counts.values()))

    if denominator == 0:
        return None

    return Fraction(numerator, denominator, _normalize=False)


def corpus_bleu(list_of_references, hypotheses, weights=(0.34, 0.33, 0.33),
                smoothing_function=None, auto_reweigh=False,
                emulate_multibleu=False):
    # Before proceeding to compute BLEU, perform sanity checks.

    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), \
            "The number of hypotheses and their reference(s) should be the same"

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        assert type(hypothesis.ngram) == dict and \
                all(type(reference.ngram) == dict for reference in references)
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision_amr(references, hypothesis, i)
            if p_i == None:
                continue
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len =  hypothesis.length
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length_amr(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        max_gram = max([x for x,y in p_denominators.items() if y > 0])
        if max_gram < len(weights):
            weights = ( 1.0 / max_gram ,) * max_gram
            print('Auto_reweigh, max-gram is', max_gram, 'new weight is', weights)

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis,
                             hyp_len=hyp_len, emulate_multibleu=emulate_multibleu)
    s = (w * math.log(p_i) for i, (w, p_i) in enumerate(list(zip(weights, p_n))))
    s =  bp * math.exp(math.fsum(s))
    return round(s, 4) if emulate_multibleu else s

def sentence_bleu(references, hypothesis, weights=(0.34, 0.33, 0.33),
                  smoothing_function=None, auto_reweigh=False,
                  emulate_multibleu=False):
    return corpus_bleu([references], [hypothesis],
                        weights, smoothing_function, auto_reweigh,
                        emulate_multibleu)

def convert_amr_to_ngram(amr_str, max_ngrams=2):

    amr = AMRGraph(amr_str.strip())
    amr.revert_of_edges()
    ngrams = amr.extract_ngrams(max_ngrams, multi_roots=True)
    
    return ngrams
