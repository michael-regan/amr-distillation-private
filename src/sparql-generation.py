#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SPARQL generation with LLaMa
Four formats of input:
- unconstrained SPARQL relations; Input: query
- unconstrained SPARQL relationsl; Input: query + AMR 
- constrained SPARQL relations; Input: query
- constrained SPARQL relations; Input: query + AMR

"""

"""
torchrun --nproc_per_node 2 sparql-generation.py \
    --ckpt_dir ../../../models/llama/llama-2-13b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 1024 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-sparql-prompts-2023-08-14.json \
    --report_path ~/reports/sparql-qald9-13b-chat_2023-08-14.json
"""

import sys

import datetime
import fire
import json
import math
import numpy as np
import pandas as pd
import re

from ast import literal_eval

from llama import Llama

from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Optional


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_dbpedia_properties():

    # Setting DBPedia endpoint
    print("getting valid dbpedia props")
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    rel_query = """SELECT DISTINCT ?pred WHERE {
    ?pred a rdf:Property
    }
    ORDER BY ?pred"""

    sparql.setQuery(rel_query)
    rel_results = sparql.queryAndConvert()

    all_dbpedia_props = [p['pred']['value'].replace('http://dbpedia.org/ontology/','') for p in rel_results['results']['bindings']]

    print(f"# of valid dbpedia props: {len(all_dbpedia_props)}")

    return set(all_dbpedia_props)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_path: str,
    report_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    num_chunks: int = 4,
    max_gen_len: Optional[int] = None
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size    
    )

    # Setting DBPedia endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    with open(data_path, 'r') as fin:
        messages = json.load(fin)

    theseMessages = [i['messages'] for i in messages]

    theseMessageInstanceChunks = chunks(messages, num_chunks)
    theseMessageChunks = chunks(theseMessages, num_chunks)

    theseResults = list()

    count_hallucinations, count_valid_rels, count_detected_hallucinations, count_error_verification = 0,0,0,0
    
    total_results, total_malformed = 0,0

    valid_dbpedia_props = get_dbpedia_properties()

    for messageInstanceChunk, messageChunk in zip(theseMessageInstanceChunks, theseMessageChunks):

        results = generator.chat_completion(
            messageChunk,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for d, result in zip(messageInstanceChunk, results):

            message = d['messages']

            if len(d['rels_to_include'])>0:
                valid_dbpedia_props = d['rels_to_include']
         

            print(f"Question: {d['question']}")
            print(f"EN Question: {d['en_question']}")

            rc = result['generation']['content'].strip()

            if rc[0]!='{' or rc[-1]!='}':
                print("Error in generation, cleaning up...")
                print(rc)
                if '{' not in rc or '}' not in rc:
                    d['score'] = 'Error in generation'
                    d['content'] = result['generation']['content']
                    d['temperature'] = temperature
                    theseResults.append(d)
                    continue
                else:
                    rc, sep, tail = rc.partition('}')
                    rc += '}'
                    print(rc)
            
            try:
                literal_results = literal_eval(rc)

                print(
                    f"> Hypothesis: {literal_results}"
                )

                hyp_sparql = literal_results['sparql_query']
                hyp_relations = literal_results['relations']
                hyp_verification = literal_results['verification']

                for hyp_rel, hyp_ver in zip(hyp_relations, hyp_verification):
                    if hyp_ver and hyp_rel not in valid_dbpedia_props:
                        count_hallucinations+=1
                    elif hyp_ver and hyp_rel in valid_dbpedia_props:
                        count_valid_rels+=1
                    elif not hyp_ver and hyp_rel not in valid_dbpedia_props:
                        count_detected_hallucinations+=1
                    else:
                        count_error_verification+=1


                ref_sparql = d['gold_sparql']

                print(
                    f"> Reference: {ref_sparql}"
                )
            except Exception as e:
                print(f"Error in literal_eval: {e}")
                error_note = 'Error in literal_eval'

            try:
                sparql.setQuery(hyp_sparql)
                sparql_results = sparql.query().convert()
                pattern = r'<results distinct="false" ordered="true">\s*</results>'
                match = re.search(pattern, sparql_results.toxml())

                if not match:
                    print("GOT RESULT")
                    total_results += 1

            except Exception as e:
                print(f"Malformed query: {e}")
                total_malformed += 1

            print("\n==================================\n")

            d['content'] = result['generation']['content']
            d['temperature'] = temperature
            theseResults.append(d)

    print(f"Total matched results: {total_results}")
    print(f"Total malformed queries: {total_malformed}")
    print()
    print(f"Count hallucinations: {count_hallucinations}")
    print(f"Count detected hallus: {count_detected_hallucinations}")
    print(f"Count valid predicted rels: {count_valid_rels}")
    print(f"Count errors in verification: {count_error_verification}")
    print()

    # print(f"Write report to: {report_path}")
    # with open(report_path, 'w') as fout:
    #     json.dump(theseResults, fout, indent=4)

    # # quick view of results

    # language = '-'.join(theseResults[0]['id'].split('-')[1:])

    # ckpt_name = ckpt_dir.split('/')[-1]
    # final_results_dict = analyze_results(theseResults, language, ckpt_name)

    # print(f"Adding compiled results to: {compiled_results_path}")

    # with jsonlines.open(compiled_results_path, 'a') as writer:
    #     writer.write(final_results_dict)

if __name__ == "__main__":
    fire.Fire(main)