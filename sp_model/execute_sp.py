import argparse
import re
import json
import os
import torch
from datasets import load_metric
import sys
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sp_search.loading_utils import load_paraphrase_model, load_compression_model, load_fusion_model
from sp_search.rouge_scoring import compute_rouge
from sp_search.summarization_program import Node, Tree, draw_sp

metric = load_metric("rouge", cache_dir="../cache", seed=42)

torch_device_fusion = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_device_compression = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_device_paraphrase = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_program_with_leaves(input_text):
    tokens = input_text.split(" ")
    sentence_tokens = {}
    leaves_program = ""
    for token in tokens:
        if re.match("^<S([0-9]|[1-9][0-9]|[1-9][0-9][0-9])>$", token):
            sentence_tokens[token] = True

    sentence_tokens = sorted(sentence_tokens.keys(), key=lambda x: int(x[2:-1]))

    for (i, sentence) in enumerate(sentence_tokens):
        leaves_program += "[ " + sentence.strip() + " ] "

    return leaves_program.strip()


def get_source_documents_and_gold_summaries(document_file):
    source_documents, gold_summaries = {}, {}
    with open(document_file, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
        for (i, line) in enumerate(lines):
            gold_summaries[i] = ""
            source_documents[i] = []
            parts = line.split("\t")
            document_sents = parts[0].split("|||")
            gold_summaries[i] = parts[1].strip()
            assert len(document_sents) > 0
            source_documents[i] = document_sents

    return source_documents, gold_summaries


def execute_operation(args, model, tokenizer, sent1, sent2=None):
    if sent2 is not None:
        batch_tokens = tokenizer([sent1 + " </s> " + sent2], truncation=True, padding='longest',
                                 max_length=args.max_src_length,
                                 return_tensors="pt").to(torch_device_fusion)
        generated = model.generate(**batch_tokens, max_length=args.max_target_length, num_beams=args.num_beams,
                                   num_return_sequences=args.num_return_sequences, temperature=args.temperature)
    else:
        batch_tokens = tokenizer([sent1], truncation=True, padding='longest',
                                 max_length=args.max_src_length,
                                 return_tensors="pt").to(torch_device_fusion)
        generated = model.generate(**batch_tokens, max_length=args.max_target_length, num_beams=args.num_beams,
                                   num_return_sequences=args.num_return_sequences, length_penalty=2.0,
                                   repetition_penalty=2.0)

    return tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


def get_search_program_and_summaries(search_file, gold_summaries):
    search_program_summaries, search_program_strings, search_program_strings_with_intermediates = \
        OrderedDict(), OrderedDict(), OrderedDict()
    with open(search_file, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
        for (i, line) in enumerate(lines):
            parts = line.split("\t")
            index = int(parts[0])
            search_program_summaries[index] = parts[2].strip()
            search_program_strings_with_intermediates[index] = parts[3].strip()
            search_program_strings[index] = parts[4].strip()

    for index, gold_summary in gold_summaries.items():
        if index not in search_program_strings:
            search_program_summaries[index] = ""
            search_program_strings_with_intermediates[index] = ""
            search_program_strings[index] = ""
    return search_program_summaries, search_program_strings, search_program_strings_with_intermediates


def generate_summary_from_program(program_string,
                                  sents,
                                  fusion_tokenizer=None,
                                  fusion_model=None,
                                  compression_tokenizer=None,
                                  compression_model=None,
                                  paraphrase_tokenizer=None,
                                  paraphrase_model=None,
                                  should_generate=False,
                                  input_text=None,
                                  program_string_with_intermediates=None):
    program_string = program_string.replace("] (", "] [ (")
    program_string = program_string.replace("))", ") )")

    program_string = program_string[1:-1].split("[")
    program_string_with_intermediates = program_string_with_intermediates[1:-1].split(
        "] [") if program_string_with_intermediates is not None else None
    program = []  # This saves the program data structure for rendering
    summary = ""
    for (i, tree_string) in enumerate(program_string):
        tree_string = tree_string.strip()
        if tree_string.endswith("]"):
            tree_string = tree_string[:-1].strip()
        intermediates = program_string_with_intermediates[i].split(" | ")[
                        1:] if program_string_with_intermediates is not None else None
        if tree_string.count("(") != tree_string.count(")"):
            print("Problem with bracket")
            return None, None
        operation_stack, operator_stack = [], []
        tokens = tree_string.split(" ")
        intermediate_index, intermediate_node_index = 0, 1
        for token in tokens:
            token = token.strip()
            if token == "(" or token == "":
                continue
            elif token == "fusion" or token == "compression" or token == "paraphrase":
                operation_stack.append(token)
            elif re.match("^<S([0-9]|[1-9][0-9]|[1-9][0-9][0-9])>$", token):
                index = int(token[2:-1])
                assert index < len(sents), f"Index {index} should be within the list of indices {len(sents)}"
                node = Node(operation="",
                            sentence=sents[index],
                            index=index,
                            score=-1,
                            height=1)
                operator_stack.append(node)
            elif token == ")":
                if len(operation_stack) == 0:
                    continue
                curr_operation = operation_stack.pop()
                if curr_operation == "fusion":
                    if len(operator_stack) < 2:
                        print("Problem with fusion")
                        return None, None
                    operator2_node = operator_stack.pop()
                    operator1_node = operator_stack.pop()
                    if not should_generate:
                        generation = "dummy"
                    else:
                        if intermediates is None:
                            generation = execute_operation(args, fusion_model, fusion_tokenizer,
                                                           operator1_node.sentence,
                                                           operator2_node.sentence)
                        else:
                            generation = intermediates[intermediate_index].split(" --> ")[1].strip()
                            intermediate_index += 1

                    node = Node(operation="fusion",
                                sentence=generation,
                                index=intermediate_node_index,
                                score=-1,
                                height=max(operator1_node.height, operator2_node.height) + 1)
                    intermediate_node_index += 1
                    operator_stack.append(node)
                    node.left = operator1_node
                    node.right = operator2_node
                elif curr_operation == "compression":
                    if len(operator_stack) == 0:
                        print("Problem with compression")
                        return None, None
                    operator_node = operator_stack.pop()
                    if not should_generate:
                        generation = "dummy"
                    else:
                        if intermediates is None:
                            generation = execute_operation(args, compression_model, compression_tokenizer,
                                                           operator_node.sentence)
                        else:
                            generation = intermediates[intermediate_index].split(" --> ")[1].strip()
                            intermediate_index += 1

                    node = Node(operation="compression",
                                sentence=generation,
                                index=intermediate_node_index,
                                score=-1,
                                height=operator_node.height + 1)
                    intermediate_node_index += 1
                    operator_stack.append(node)
                    node.left = operator_node
                elif curr_operation == "paraphrase":
                    if len(operator_stack) == 0:
                        print("Problem with paraphrase")
                        return None, None
                    operator_node = operator_stack.pop()

                    if not should_generate:
                        generation = "dummy"
                    else:
                        if intermediates is None:
                            generation = execute_operation(args, paraphrase_model, paraphrase_tokenizer,
                                                           operator_node.sentence)
                        else:
                            generation = intermediates[intermediate_index].split(" --> ")[1].strip()
                            intermediate_index += 1

                    node = Node(operation="paraphrase",
                                sentence=generation,
                                index=intermediate_node_index,
                                score=-1,
                                height=operator_node.height + 1)
                    intermediate_node_index += 1
                    operator_stack.append(node)
                    node.left = operator_node
                else:
                    continue
            else:
                continue

        if len(operator_stack) > 0:
            program.append(Tree(operator_stack[0]))
            summary += operator_stack[0].sentence.strip() + " "

    return summary.strip(), program


def construct_string_with_intermediates(program):
    program_string_with_intermediates = ""
    for tree in program:
        tree_string_with_intermediates = tree.construct_execution_string(tree.root)
        tree_string_with_intermediates = tree_string_with_intermediates[0] + tree_string_with_intermediates[1]
        program_string_with_intermediates += "[ " + tree_string_with_intermediates + " ] "

    return program_string_with_intermediates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--document_file', default='../data/test.tsv', type=str,
                        help='Path to the document file with gold summaries')
    parser.add_argument('--search_file', default='../data/sp_search_test.tsv', type=str,
                        help='Path to the file where all the spsearch programs are saved')
    parser.add_argument('--test_file', default='../data/test.json', type=str, help='Path to processed test file')
    parser.add_argument('--sp_generations', default='../models/extract-and-build/generations_matchsum_min_length10.txt', type=str,
                        help='Path to file where predicted programs are saved')
    parser.add_argument('--sp_dir', default='../data/sp_dir', type=str,
                        help='Path to the directory where searched and predicted programs will be rendered (as pdfs)')
    parser.add_argument('--output_file', default='../data/output.tsv', type=str,
                        help='Path to the output file which saves everything (program strings, summaries, etc)')
    parser.add_argument('--should_generate', default=True, type=bool)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--num_beams', default=5, type=int)
    parser.add_argument('--max_src_length', default=60, type=int)
    parser.add_argument('--max_target_length', default=60, type=int)
    parser.add_argument('--temperature', default=1.5, type=float)
    parser.add_argument('--cache_dir', default='../cache', type=str)
    parser.add_argument('--paraphrase_model_path', default='../modules/paraphrase', type=str)
    parser.add_argument('--compression_model_path', default='../modules/compression', type=str)
    parser.add_argument('--fusion_model_path', default='../modules/fusion', type=str)

    args = parser.parse_args()

    if args.should_generate:
        compression_model, compression_tokenizer = load_compression_model(args.compression_model_path,
                                                                          args.cache_dir,
                                                                          torch_device_compression)
        fusion_model, fusion_tokenizer = load_fusion_model(args.fusion_model_path,
                                                           args.cache_dir,
                                                           torch_device_fusion)
        paraphrase_model, paraphrase_tokenizer = load_paraphrase_model(args.paraphrase_model_path,
                                                                       args.cache_dir,
                                                                       torch_device_paraphrase)
        output = open(args.output_file, "w", encoding="utf-8-sig")
        # Save everything in one file
        output.write("Index\tGold Summary\tsearch_program_string\tsearch_program_string_with_intermediates\t"
                     "search_program_summary\tsearch_leaves\t"
                     "search_leaves_summary\tpredicted_program_string\tpredicted_program_string_with_intermediates\t"
                     "predicted_program_summary\tpredicted_leaves\tpredicted_leaves_summary\n")
    else:
        compression_model, compression_tokenizer, fusion_model, fusion_tokenizer, paraphrase_model, \
        paraphrase_tokenizer = None, None, None, None, None, None

    source_documents, gold_summaries = get_source_documents_and_gold_summaries(args.document_file)

    test_lines = open(args.test_file, "r", encoding="utf-8-sig").read().splitlines()
    predicted_program_strings = []
    for line in open(args.sp_generations, "r", encoding="utf-8-sig"):
        predicted_program_strings.append(line.split("\t")[1])

    search_program_summaries, search_program_strings, search_program_strings_with_intermediates = \
        get_search_program_and_summaries(args.search_file, gold_summaries)

    test_lines = test_lines[:len(predicted_program_strings)]
    assert len(test_lines) == len(predicted_program_strings)

    well_formed_count = 0
    all_gold_summaries, all_search_program_summaries, all_search_leaves_summaries, all_predicted_program_summaries, \
    all_predicted_leaves_summaries = [], [], [], [], []
    all_search_programs, all_predicted_programs = [], []
    for i, (test_line, predicted_program_string) in enumerate(zip(test_lines, predicted_program_strings)):
        print(i)
        gold_summary = gold_summaries[i]
        search_program_string = search_program_strings[i]
        search_program_string_with_intermediates = search_program_strings_with_intermediates[i]
        search_program_summary = search_program_summaries[i]

        test_line = json.loads(test_line)
        assert i in source_documents

        temp_search_program_summary, search_program = \
            generate_summary_from_program(search_program_string,
                                          source_documents[i],
                                          program_string_with_intermediates=search_program_string_with_intermediates,
                                          should_generate=False
                                          )
        all_search_programs.append(search_program)

        search_leaves = get_program_with_leaves(search_program_string)
        search_leaves_summary, _ = generate_summary_from_program(search_leaves,
                                                                 source_documents[i],
                                                                 fusion_tokenizer=fusion_tokenizer,
                                                                 fusion_model=fusion_model,
                                                                 compression_tokenizer=compression_tokenizer,
                                                                 compression_model=compression_model,
                                                                 paraphrase_tokenizer=paraphrase_tokenizer,
                                                                 paraphrase_model=paraphrase_model,
                                                                 should_generate=args.should_generate)

        predicted_leaves = get_program_with_leaves(test_line["text"])
        predicted_leaves_summary, _ = generate_summary_from_program(predicted_leaves,
                                                                    source_documents[i],
                                                                    fusion_tokenizer=fusion_tokenizer,
                                                                    fusion_model=fusion_model,
                                                                    compression_tokenizer=compression_tokenizer,
                                                                    compression_model=compression_model,
                                                                    paraphrase_tokenizer=paraphrase_tokenizer,
                                                                    paraphrase_model=paraphrase_model,
                                                                    should_generate=args.should_generate,
                                                                    input_text=test_line["text"])

        predicted_program_summary, predicted_program = generate_summary_from_program(predicted_program_string,
                                                                                     source_documents[i],
                                                                                     fusion_tokenizer=fusion_tokenizer,
                                                                                     fusion_model=fusion_model,
                                                                                     compression_tokenizer=compression_tokenizer,
                                                                                     compression_model=compression_model,
                                                                                     paraphrase_tokenizer=paraphrase_tokenizer,
                                                                                     paraphrase_model=paraphrase_model,
                                                                                     should_generate=args.should_generate,
                                                                                     input_text=test_line["text"])

        if predicted_program_summary:
            predicted_program_string_with_intermediates = construct_string_with_intermediates(predicted_program)
            all_predicted_programs.append(predicted_program)
            networkx_graph_search = draw_sp(search_program)
            networkx_graph_predicted = draw_sp(predicted_program)
            try:
                file_path = os.path.join(args.sp_dir, "program_search_" + str(i))
                networkx_graph_search.render(file_path, view=False)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            try:
                file_path = os.path.join(args.sp_dir, "program_predicted_" + str(i))
                networkx_graph_predicted.render(file_path, view=False)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            well_formed_count += 1
        else:
            predicted_program_string_with_intermediates = ""
            all_predicted_programs.append([])
            predicted_program_summary = ""

        if args.should_generate:
            all_gold_summaries.append(gold_summaries[i])
            all_search_program_summaries.append(search_program_summary)
            all_search_leaves_summaries.append(search_leaves_summary)
            all_predicted_program_summaries.append(predicted_program_summary)
            all_predicted_leaves_summaries.append(predicted_leaves_summary)

            output.write(
                f"{i}\t{gold_summary}\t{search_program_string}\t{search_program_string_with_intermediates}"
                f"\t{search_program_summary}\t{search_leaves}"
                f"\t{search_leaves_summary}\t{predicted_program_string}\t{predicted_program_string_with_intermediates}"
                f"\t{predicted_program_summary}\t{predicted_leaves}\t{predicted_leaves_summary}\n")
            output.flush()
            os.fsync(output.fileno())

    if args.should_generate:
        output.write('------------- Evaluation Metrics -------------\n\n')
        overall_rouge_scores_predicted_programs = compute_rouge(all_gold_summaries, all_predicted_program_summaries)

        print(f'Predicted Program Rouge1 Score = {overall_rouge_scores_predicted_programs["rouge1"]}\n')
        print(f'Predicted Program Rouge2 Score = {overall_rouge_scores_predicted_programs["rouge2"]}\n')
        print(f'Predicted Program RougeL Score = {overall_rouge_scores_predicted_programs["rougeL"]}\n')
        print(f'Predicted Program RougeLsum Score = {overall_rouge_scores_predicted_programs["rougeLsum"]}\n\n')

        overall_rouge_scores_predicted_leaves = compute_rouge(all_gold_summaries, all_predicted_leaves_summaries)

        print(f'Predicted Leaves Rouge1 Score = {overall_rouge_scores_predicted_leaves["rouge1"]}\n')
        print(f'Predicted Leaves Rouge2 Score = {overall_rouge_scores_predicted_leaves["rouge2"]}\n')
        print(f'Predicted Leaves RougeL Score = {overall_rouge_scores_predicted_leaves["rougeL"]}\n')
        print(f'Predicted Leaves RougeLsum Score = {overall_rouge_scores_predicted_leaves["rougeLsum"]}\n\n')

        overall_rouge_scores_search_programs = compute_rouge(all_gold_summaries, all_search_program_summaries)

        print(f'Search Program Rouge1 Score = {overall_rouge_scores_search_programs["rouge1"]}\n')
        print(f'Search Program Rouge2 Score = {overall_rouge_scores_search_programs["rouge2"]}\n')
        print(f'Search Program RougeL Score = {overall_rouge_scores_search_programs["rougeL"]}\n')
        print(f'Search Program RougeLsum Score = {overall_rouge_scores_search_programs["rougeLsum"]}\n\n')

        overall_rouge_scores_search_leaves = compute_rouge(all_gold_summaries, all_search_leaves_summaries)

        print(f'Search Leaves Rouge1 Score = {overall_rouge_scores_search_leaves["rouge1"]}\n')
        print(f'Search Leaves Rouge2 Score = {overall_rouge_scores_search_leaves["rouge2"]}\n')
        print(f'Search Leaves RougeL Score = {overall_rouge_scores_search_leaves["rougeL"]}\n')
        print(f'Search Leaves RougeLsum Score = {overall_rouge_scores_search_leaves["rougeLsum"]}\n\n')

        output.write(f'Well-formed accuracy = {well_formed_count / len(predicted_program_strings)}')

    print(f'Well-formed accuracy = {well_formed_count / len(predicted_program_strings)}')
