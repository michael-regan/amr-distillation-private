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
torchrun --nproc_per_node 1 sparql-generation.py \
    --ckpt_dir ../../../models/llama/llama-2-7b-chat \
    --tokenizer_path ../../../models/llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_batch_size 4 \
    --num_chunks 2 \
    --temperature 0.9 \
    --data_path ~/portfolio/amr-distillation-private/data/llama-sparql-prompts-2023-08-15.json \
    --report_path ~/reports/sparql-qald9-7b-chat_8exs_2023-08-15.json
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

from SPARQLWrapper import SPARQLWrapper, JSON, XML
from typing import Optional


# Setting DBPedia endpoint
sparql = SPARQLWrapper("http://dbpedia.org/sparql")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_dbpedia_properties():

    # Setting DBPedia endpoint
    print("getting valid dbpedia props")
    sparql.setReturnFormat(JSON)
    
    rel_query = """SELECT DISTINCT ?pred WHERE {
    ?pred a rdf:Property
    }
    ORDER BY ?pred"""

    # rel_query = """SELECT ?pred (COUNT(?pred) as ?pCount)
    # WHERE
    # {
    # ?s ?pred ?o .
    # }
    # GROUP BY ?pred
    # HAVING (COUNT(?pred)>3)
    # """

    sparql.setQuery(rel_query)
    rel_results = sparql.queryAndConvert()

    all_dbpedia_props = [p['pred']['value'].replace('http://dbpedia.org/ontology/','') for p in rel_results['results']['bindings']]

    #all_dbpedia_props = [p.lower() for p in all_dbpedia_props]
    print(f"# of valid dbpedia props: {len(all_dbpedia_props)}")

    return set(all_dbpedia_props)


def verify_exist_dbpedia_obj(page):

    sparql.setReturnFormat(JSON)

    page_pa = page.split('/')

    tgt_name = page_pa[-1]
    tgt_onto = page_pa[-2]

    if tgt_onto == 'page':
        tgt_onto = 'dbp'
    elif tgt_onto == 'resource':
        tgt_onto = 'dbr'
        
    else:
        print("Unknown ontology")
        
    q = f"""ASK {{
        VALUES (?r) {{ ({tgt_onto}:{tgt_name}) }}
            {{ ?r ?p ?o }}
            UNION
            {{ ?s ?r ?o }}
            UNION
            {{ ?s ?p ?r }}
        }}"""
        
    sparql.setQuery(q)
    results = sparql.query().convert()

    return results['boolean']
    

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

    with open(data_path, 'r') as fin:
        messages = json.load(fin)

    theseMessages = [i['messages'] for i in messages]

    theseMessageInstanceChunks = chunks(messages, num_chunks)
    theseMessageChunks = chunks(theseMessages, num_chunks)

    theseResults = list()

    count_results_dict = {
        "unconstrained": {'count_hallucinations': 0, 
                          'count_valid_rels':0, 
                          'count_detected_hallucinations':0,
                          'count_rels_false_positives': 0,
                          'total_hypotheses': 0,
                          'total_queries': 0,
                          'total_answers_existing': 0,
                          'total_answers_pred_same_as_returned': 0,
                          'total_answers_pred_same_as_known': 0,
                          'total_answers': 0,
                          'non_dbpedia_answer': 0},
        "unconstrained_amr": {'count_hallucinations': 0, 
                          'count_valid_rels':0, 
                          'count_detected_hallucinations':0,
                          'count_rels_false_positives': 0,
                          'total_hypotheses': 0,
                          'total_queries': 0,
                          'total_answers_existing': 0,
                          'total_answers_pred_same_as_returned': 0,
                          'total_answers_pred_same_as_known': 0,
                          'total_answers': 0,
                          'non_dbpedia_answer': 0},
        "constrained": {'count_hallucinations': 0, 
                          'count_valid_rels':0, 
                          'count_detected_hallucinations':0,
                          'count_rels_false_positives': 0,
                          'total_hypotheses': 0,
                          'total_queries': 0,
                          'total_answers_existing': 0,
                          'total_answers_pred_same_as_returned': 0,
                          'total_answers_pred_same_as_known': 0,
                          'total_answers': 0,
                          'non_dbpedia_answer': 0},
        "constrained_amr":{'count_hallucinations': 0, 
                          'count_valid_rels':0, 
                          'count_detected_hallucinations':0,
                          'count_rels_false_positives': 0,
                          'total_hypotheses': 0,
                          'total_queries': 0,
                          'total_answers_existing': 0,
                          'total_answers_pred_same_as_returned': 0,
                          'total_answers_pred_same_as_known': 0,
                          'total_answers': 0,
                          'non_dbpedia_answer': 0}
    }

    def get_results_dict(d):

        if d['manipulated']==0 and d['include_amr']==False:
            thisDict = count_results_dict['unconstrained']
        elif d['manipulated']==0 and d['include_amr']==True:
            thisDict = count_results_dict['unconstrained_amr']
        elif d['manipulated']==1 and d['include_amr']==False:
            thisDict = count_results_dict['constrained']
        elif d['manipulated']==1 and d['include_amr']==True:
            thisDict = count_results_dict['constrained_amr']

        return thisDict
        
    
    total_results, total_malformed, total_literal_eval_errors = 0,0,0

    valid_dbpedia_props = get_dbpedia_properties()

    sparql.setReturnFormat(XML)

    for messageInstanceChunk, messageChunk in zip(theseMessageInstanceChunks, theseMessageChunks):

        results = generator.chat_completion(
            messageChunk,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for d, result in zip(messageInstanceChunk, results):

            results_summary = ''

            message = d['messages']

            qald9_answers = d['qald9_answers']

            if len(d['rels_to_include'])>0:
                valid_dbpedia_props = d['rels_to_include']
        
            print(f"Question: {d['question']}")
            #print(f"EN Question: {d['en_question']}")

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
                    #rc, sep, tail = rc.partition('}')

                    rc = '}'.join(rc.split('}')[:-1]) + '}'
                    print(rc)

            try:
                rc = rc.replace('[[', '[').replace(']]', ']')
                literal_results = literal_eval(rc)

            except Exception as e:
                print(f"Error in literal_eval: {e}")
                results_summary += 'Error in literal_eval'
                total_literal_eval_errors+=1
                d['content'] = result['generation']['content']
                d['temperature'] = temperature
                d['results_summary'] = results_summary
                theseResults.append(d)
                continue

            print(
                f"> Hypothesis: {literal_results}"
            )
            print()
            print(
                f"> Hyp query: {literal_results['sparql_query']}"
            )
            print()
            print(
                f"> Hyp relations: {literal_results['relations']}"
            )
            print()
            print(
                f"> Hyp answers: {literal_results['answers']}"
            )


            hyp_sparql = literal_results['sparql_query']
            hyp_relations = literal_results['relations']
            hyp_verification = literal_results['verification']

            thisDict = get_results_dict(d)

            for hyp_rel, hyp_ver in zip(hyp_relations, hyp_verification):
                if hyp_ver and hyp_rel not in valid_dbpedia_props:
                    thisDict['count_hallucinations']+=1
                    print(f"Hallucination: {hyp_rel}")
                    results_summary += f"Hallucination: {hyp_rel}\n"
                elif hyp_ver and hyp_rel in valid_dbpedia_props:
                    thisDict['count_valid_rels']+=1
                    results_summary += f"Valid relation: {hyp_rel}\n"
                elif not hyp_ver and hyp_rel not in valid_dbpedia_props:
                    thisDict['count_detected_hallucinations']+=1
                    results_summary += f"Detected hallucination: {hyp_rel}\n"
                elif not hyp_ver and hyp_rel in valid_dbpedia_props:
                    thisDict['count_rels_false_positives']+=1
                    results_summary += f"False positive: {hyp_rel}\n"

            thisDict['total_hypotheses'] += len(hyp_relations)

            thisDict['total_queries'] += 1

                # verify existence in DBPedia of answers
            for hyp_ans in literal_results['answers']:
                if type(hyp_ans)==str:
                    if 'dbpedia.org' in hyp_ans:

                        try:
                            if verify_exist_dbpedia_obj(hyp_ans):
                                thisDict['total_answers_existing'] += 1
                                print(f"Verified to exist in DBPedia: {hyp_ans}")
                                results_summary += f"Verified to exist in DBPedia: {hyp_ans}\n"

                        except Exception as e:
                            print(f"Malformed DBPedia verification: {e}")
                            results_summary += f"Malformed DBPedia verification: {e}"
                            total_malformed += 1

                        thisDict['total_answers'] += 1
                    else:
                        thisDict['non_dbpedia_answer']
                else:
                    pass

            ref_sparql = d['gold_sparql']

            print(
                f"> Reference: {ref_sparql}"
            )


            try:
                hyp_sparql = literal_results['sparql_query']
                sparql.setReturnFormat(XML)
                sparql.setQuery(hyp_sparql)
                sparql_results = sparql.query().convert()
                pattern = r'<results distinct="false" ordered="true">\s*</results>'
                match = re.search(pattern, sparql_results.toxml())

                #thisDict = get_results_dict(d)

                if not match:
                    print("GOT RESULT")
                    print(sparql_results.toxml())
                    total_results += 1

                    # verify correctness
                    for hyp_ans in literal_results['answers']:
                        if type(hyp_ans)==str:
                            if 'dbpedia.org' in hyp_ans:
                                if hyp_ans in sparql_results.toxml():
                                    thisDict['total_answers_pred_same_as_returned'] += 1
                                    print(f"ANSWER predicted matches that returned by SPARQL: {hyp_ans}")
                                    results_summary += f"ANSWER predicted matches that returned by SPARQL: {hyp_ans}\n"
                                    

            except Exception as e:
                print(f"Malformed query: {e}")
                results_summary += f"Malformed query: {e}"
                total_malformed += 1


            try:
                # check if answers match qald9 answers
                #thisDict = get_results_dict(d)
                for ha in literal_results['answers']:

                    if type(ha)==bool and ha in qald9_answers:
                        print(f"Predicted bool answer matchd with QALD-9: {ha}")
                        results_summary += f"Predicted bool answer matchd with QALD-9: {ha}\n"
                        thisDict['total_answers_pred_same_as_known'] += 1

                    elif ha in qald9_answers or ha.replace('page/', 'resource/') in qald9_answers or ha.replace('resource/', 'page/') in qald9_answers or ha.replace('https', 'http') in qald9_answers:
                        print(f"Predicted answer matchd with QALD-9: {ha}")
                        results_summary += f"Predicted answer matchd with QALD-9: {ha}\n"
                        thisDict['total_answers_pred_same_as_known'] += 1

                    elif ha.replace('page/', 'resource/').replace('https', 'http') in qald9_answers:
                        print(f"Predicted answer matchd with QALD-9: {ha}")
                        results_summary += f"Predicted answer matchd with QALD-9: {ha}\n"
                        thisDict['total_answers_pred_same_as_known'] += 1

            except Exception as e:
                print(f"Error matching answers to qald9: {e}")
                results_summary += f"Error matching answers to qald9: {e}"

            print("\n==================================\n")

            d['content'] = result['generation']['content']
            d['temperature'] = temperature
            d['results_summary'] = results_summary
            theseResults.append(d)

            if len(theseResults)%50==0:
                print("ongoing results")
                print(count_results_dict)
                print()
                print("\n==================================\n")

    print()
    print(f"Total matched results: {total_results}")
    print(f"Total malformed queries: {total_malformed}")
    print(f"Total literal_eval errors: {total_literal_eval_errors}")
    print()
    print("\n==================================\n")
    print()
    for k, v in count_results_dict.items():
        print(k)
        print(v)
        print()

    theseResultsDict = {'results': theseResults, 'count_results': count_results_dict}

    # print("Unconstrained")
    # print(f"Count hallucinations: {count_results_dict['unconstrained']['count_hallucinations']}")
    # print(f"Count detected hallus: {['count_detected_hallucinations']}")
    # print(f"Count valid predicted rels: {['count_valid_rels']}")
    # print(f"Count errors in verification: {['count_error_verification']}")
    # print()

    print(f"Write report to: {report_path}")
    with open(report_path, 'w', encoding='utf8') as fout:
        json.dump(theseResultsDict, fout, ensure_ascii=False, indent=4)

    # # quick view of results

    # language = '-'.join(theseResults[0]['id'].split('-')[1:])

    # ckpt_name = ckpt_dir.split('/')[-1]
    # final_results_dict = analyze_results(theseResults, language, ckpt_name)

    # print(f"Adding compiled results to: {compiled_results_path}")

    # with jsonlines.open(compiled_results_path, 'a') as writer:
    #     writer.write(final_results_dict)

if __name__ == "__main__":
    fire.Fire(main)