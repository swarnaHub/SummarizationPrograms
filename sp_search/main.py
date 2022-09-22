import nltk
import os
import numpy as np
import torch
import timeit
import argparse
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sp_search.summarization_program import Node, Tree, draw_sp
from sp_search.loading_utils import load_paraphrase_model, load_compression_model, load_fusion_model
from sp_search.rouge_scoring import compute_rouge


def retrieve_operation(args, sent1, summary_sent, saved_operations, operation, sent2=None):
    assert (sent1, sent2, operation) in saved_operations, "Something wrong, the operation result should be saved"
    tgt_texts = saved_operations[(sent1, sent2, operation)]

    rouge_scores = compute_rouge([summary_sent] * len(tgt_texts), tgt_texts, use_agregator=False)[args.score_metric]
    rouge_scores = np.array([score.fmeasure * 100 for score in rouge_scores])
    best_index = np.argmax(rouge_scores, axis=0)

    return tgt_texts[best_index], rouge_scores[best_index]


def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def execute_batched_operations(args,
                               queue,
                               fusion_model,
                               fusion_tokenizer,
                               compression_model,
                               compression_tokenizer,
                               paraphrase_model,
                               paraphrase_tokenizer,
                               all_sentences,
                               saved_operations):
    compression_elements, paraphrase_elements, fusion_elements = [], [], []
    for element in queue:
        if element[4] == "fusion" and (all_sentences[element[1]], all_sentences[element[2]], "fusion") not in saved_operations:
            fusion_elements.append(element)
        elif element[4] == "compression" and (all_sentences[element[1]], None, "compression") not in saved_operations:
            compression_elements.append(element)
        elif element[4] == "paraphrase" and (all_sentences[element[1]], None, "paraphrase") not in saved_operations:
            paraphrase_elements.append(element)

    for batch in get_batch(fusion_elements, int(args.batch_size / 2)):
        batch_elements = [all_sentences[element[1]] + " </s> " + all_sentences[element[2]] for element in batch]
        batch_elements.extend([all_sentences[element[2]] + " </s> " + all_sentences[element[1]] for element in batch])
        batch_tokens = fusion_tokenizer(batch_elements, truncation=True, padding='longest',
                                        max_length=256, return_tensors="pt").to(torch_device_fusion)
        generated = fusion_model.generate(**batch_tokens, max_length=args.max_target_length, num_beams=args.num_beams,
                                          num_return_sequences=args.num_return_sequences, length_penalty=2.0,
                                          repetition_penalty=2.0)
        tgt_texts = fusion_tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
        for i in range(len(batch)):
            saved_operations[(all_sentences[batch[i][1]], all_sentences[batch[i][2]], "fusion")] = \
                tgt_texts[i * args.num_return_sequences: (i + 1) * args.num_return_sequences]
            temp_count = args.num_return_sequences * len(batch)
            saved_operations[(all_sentences[batch[i][2]], all_sentences[batch[i][1]], "fusion")] = \
                tgt_texts[temp_count + i * args.num_return_sequences: temp_count + (i + 1) * args.num_return_sequences]

    for batch in get_batch(compression_elements, args.batch_size):
        batch_elements = [all_sentences[element[1]] for element in batch]
        batch_tokens = compression_tokenizer(batch_elements, truncation=True, padding='longest',
                                             max_length=args.max_src_length, return_tensors="pt").to(
            torch_device_compression)
        generated = compression_model.generate(**batch_tokens, max_length=args.max_target_length,
                                               num_beams=args.num_beams,
                                               num_return_sequences=args.num_return_sequences,
                                               temperature=args.temperature)
        tgt_texts = compression_tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)
        for i in range(len(batch)):
            saved_operations[(all_sentences[batch[i][1]], None, "compression")] = \
                tgt_texts[i * args.num_return_sequences: (i + 1) * args.num_return_sequences]

    for batch in get_batch(paraphrase_elements, args.batch_size):
        batch_elements = [all_sentences[element[1]] for element in batch]
        batch_tokens = paraphrase_tokenizer(batch_elements, truncation=True, padding='longest',
                                            max_length=args.max_src_length, return_tensors="pt").to(
            torch_device_paraphrase)
        generated = paraphrase_model.generate(**batch_tokens, max_length=args.max_target_length,
                                              num_beams=args.num_beams,
                                              num_return_sequences=args.num_return_sequences,
                                              temperature=args.temperature)
        tgt_texts = paraphrase_tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
        for i in range(len(batch)):
            saved_operations[(all_sentences[batch[i][1]], None, "paraphrase")] = \
                tgt_texts[i * args.num_return_sequences: (i + 1) * args.num_return_sequences]


def construct_tree(parent_child_score_map, top_indices, all_sentences):
    top_trees = []
    for index in top_indices:
        children = parent_child_score_map[index]
        root = Node(operation=children[3], sentence=all_sentences[index], index=index, score=children[2], height=children[4])
        tree = Tree(root)

        queue = [root]
        while len(queue) > 0:
            curr_node = queue.pop(0)
            children = parent_child_score_map[curr_node.index]
            if children[0] != -1:
                child_operation = parent_child_score_map[children[0]][3]
                child_sentence = all_sentences[children[0]]
                child_index = children[0]
                child_score = parent_child_score_map[children[0]][2]
                child_height = parent_child_score_map[children[0]][4]
                assert child_height < curr_node.height, "Child's height should be less than parent's height"
                new_node = Node(operation=child_operation, sentence=child_sentence, index=child_index,
                                score=child_score,
                                height=child_height)
                curr_node.left = new_node
                queue.append(new_node)
            if children[1] != -1:
                child_operation = parent_child_score_map[children[1]][3]
                child_sentence = all_sentences[children[1]]
                child_index = children[1]
                child_score = parent_child_score_map[children[1]][2]
                child_height = parent_child_score_map[children[1]][4]
                assert child_height < curr_node.height, "Child's height should be one less than parent's height"
                new_node = Node(operation=child_operation, sentence=child_sentence, index=child_index,
                                score=child_score,
                                height=child_height)
                curr_node.right = new_node
                queue.append(new_node)

        top_trees.append(tree)

    return top_trees


def assign_top_scores(top_scores, top_indices, new_score, new_index, best_programs):
    top_scores.append(new_score)
    top_indices.append(new_index)

    sorted_top_indices = [x for _, x in sorted(zip(top_scores, top_indices), reverse=True)][:best_programs]
    sorted_top_scores = sorted(top_scores, reverse=True)[:best_programs]

    return sorted_top_scores, sorted_top_indices


def choose_fusion_order(score_1_2, score_2_1, generated_sentence_1_2, generated_sentence_2_1,
                        intermediate_to_document_sents_map, index1, index2):
    min_document_index_1, max_document_index_1 = index1, index1
    min_document_index_2, max_document_index_2 = index2, index2
    if index1 in intermediate_to_document_sents_map:
        doc_indices = intermediate_to_document_sents_map[index1]
        min_document_index_1, max_document_index_1 = min(doc_indices), max(doc_indices)

    if index2 in intermediate_to_document_sents_map:
        doc_indices = intermediate_to_document_sents_map[index2]
        min_document_index_2, max_document_index_2 = min(doc_indices), max(doc_indices)

    if max_document_index_1 <= min_document_index_2:
        return score_1_2, generated_sentence_1_2
    elif max_document_index_2 <= min_document_index_1:
        return score_2_1, generated_sentence_2_1
    elif score_1_2 > score_2_1:
        return score_1_2, generated_sentence_1_2
    else:
        return score_2_1, generated_sentence_2_1


def search_for_tree(args, document_sents, summary_sent, oracle_sent_indices,
                    compression_model, compression_tokenizer,
                    fusion_model, fusion_tokenizer,
                    paraphrase_model, paraphrase_tokenizer, saved_operations):
    queue = []  # Implemented as a max-heap that returns the element with the highest score
    parent_child_score_map = {}
    intermediate_to_document_sents_map = {}  # Saves for each intermediate generation the document sentences used in the subtree
    best_programs = args.best_programs
    top_indices = [-1] * best_programs
    top_scores = [-1] * best_programs

    # Compute rouge scores for the initial sentences (leaves)
    rouge_scores = compute_rouge([summary_sent] * len(oracle_sent_indices),
                                 [document_sents[oracle_sent_index] for oracle_sent_index in oracle_sent_indices],
                                 use_agregator=False)
    rouge_scores = rouge_scores[args.score_metric]
    scores_leaves = np.array([score.fmeasure * 100 for score in rouge_scores])

    # Each element in the queue takes the form of (heap score, sent1_index, sent2_index, height, operation_name, sentences indices used in the subtree)
    # Add initial elements to queue
    for i in range(len(oracle_sent_indices)):
        score = scores_leaves[i]
        top_scores, top_indices = assign_top_scores(top_scores, top_indices, score, oracle_sent_indices[i], best_programs)
        queue.append((score, oracle_sent_indices[i], -1, 1, "compression", []))
        queue.append((score, oracle_sent_indices[i], -1, 1, "paraphrase", []))
        parent_child_score_map[oracle_sent_indices[i]] = (-1, -1, score, "", 0)
        for j in range(i + 1, len(oracle_sent_indices)):
            queue.append((max(scores_leaves[i], scores_leaves[j]), min(oracle_sent_indices[i], oracle_sent_indices[j]),
                          max(oracle_sent_indices[i], oracle_sent_indices[j]), 1, "fusion", []))

    max_height = args.max_height
    num_trees = 1
    prev_height = 0
    all_sentences = document_sents.copy()
    while len(queue) > 0:
        # Max heap pops the element with the highest score (first element in the tuple)
        # print(f"Length of the queue = {len(queue)}")

        # When height increases, prune the queue to take the best k elements
        # Further, execute all operations
        if queue[0][3] != prev_height:
            if args.max_queue_size != -1 and len(queue) > args.max_queue_size:
                queue = sorted(queue, key=lambda tup: tup[0], reverse=True)
                queue = queue[:args.max_queue_size]
            prev_height = queue[0][3]
            execute_batched_operations(args,
                                       queue,
                                       fusion_model,
                                       fusion_tokenizer,
                                       compression_model,
                                       compression_tokenizer,
                                       paraphrase_model,
                                       paraphrase_tokenizer,
                                       all_sentences,
                                       saved_operations)

        curr_node = queue.pop(0)

        # print(f"Height = {curr_node[3]}")
        # print(f"Sentence indices to be used = {curr_node[1]} and {curr_node[2]}")
        if curr_node[3] > max_height:
            num_trees += 1
            continue

        if curr_node[4] == "fusion":
            assert curr_node[2] != -1, "Something wrong with the operation, fusion should have two arguments"
            # print("Fusion")
            # print(f"Sentences to fuse = {all_sentences[curr_node[1]]} </s> {all_sentences[curr_node[2]]}")
            generated_sentence_1_2, score_1_2 = retrieve_operation(args,
                                                                   all_sentences[curr_node[1]],
                                                                   summary_sent,
                                                                   saved_operations, "fusion",
                                                                   sent2=all_sentences[curr_node[2]])
            # print(f"Generated sentence from fusion = {generated_sentence_1_2}")

            generated_sentence_2_1, score_2_1 = retrieve_operation(args,
                                                                   all_sentences[curr_node[2]],
                                                                   summary_sent,
                                                                   saved_operations, "fusion",
                                                                   sent2=all_sentences[curr_node[1]])
            # print(f"Generated sentence from fusion = {generated_sentence_2_1}")

            score, generated_sentence = choose_fusion_order(score_1_2, score_2_1, generated_sentence_1_2,
                                                            generated_sentence_2_1, intermediate_to_document_sents_map,
                                                            curr_node[1], curr_node[2])

            child1_score, child2_score = parent_child_score_map[curr_node[1]][2], parent_child_score_map[curr_node[2]][2]
            # Ignore if the generated sentence has a worse rouge score than its children
            if score <= child1_score or score <= child2_score:
                continue

            # Check if the sentence has already been generated through some other subtree
            # Compare the height and if the current generation has a lower height, move the pointer
            if generated_sentence in all_sentences:
                previous_index = all_sentences.index(generated_sentence)
                previous_height = parent_child_score_map[previous_index][4]
                if curr_node[3] < previous_height:
                    index = previous_index
                else:
                    continue
            else:
                all_sentences.append(generated_sentence)
                index = len(all_sentences) - 1
                top_scores, top_indices = assign_top_scores(top_scores, top_indices, score, index, best_programs)

            if score == score_1_2:
                parent_child_score_map[index] = (curr_node[1], curr_node[2], score, "fusion", curr_node[3])
            else:
                parent_child_score_map[index] = (curr_node[2], curr_node[1], score, "fusion", curr_node[3])
        elif curr_node[4] == "compression":
            assert curr_node[2] == -1, "Something wrong with the operation, compression should have one argument"
            # print("Compression")
            # print(f"Sentence to compress = {all_sentences[curr_node[1]]}")
            generated_sentence, score = retrieve_operation(args,
                                                           all_sentences[curr_node[1]],
                                                           summary_sent, saved_operations,
                                                           "compression")
            # print(f"Compressed sentence = {generated_sentence}")
            child_score = parent_child_score_map[curr_node[1]][2]

            # Ignore if the generated sentence has a worse rouge score than its children
            if score <= child_score:
                continue

            # Check if the sentence has already been generated through some other subtree
            # Compare the height and if the current generation has a lower height, move the pointer
            if generated_sentence in all_sentences:
                previous_index = all_sentences.index(generated_sentence)
                previous_height = parent_child_score_map[previous_index][4]
                if curr_node[3] < previous_height:
                    index = previous_index
                else:
                    continue
            else:
                all_sentences.append(generated_sentence)
                index = len(all_sentences) - 1
                top_scores, top_indices = assign_top_scores(top_scores, top_indices, score, index, best_programs)

            parent_child_score_map[index] = (curr_node[1], -1, score, "compression", curr_node[3])
        elif curr_node[4] == "paraphrase":
            assert curr_node[2] == -1, "Something wrong with the operation, paraphrase should have one argument"
            # print("Paraphrase")
            # print(f"Sentence to paraphrase = {all_sentences[curr_node[1]]}")
            generated_sentence, score = retrieve_operation(args,
                                                           all_sentences[curr_node[1]],
                                                           summary_sent,
                                                           saved_operations,
                                                           "paraphrase")
            # print(f"Paraphrased sentence = {generated_sentence}")
            child_score = parent_child_score_map[curr_node[1]][2]

            # Ignore if the generated sentence has a worse rouge score than its children
            if score <= child_score:
                continue

            # Check if the sentence has already been generated through some other subtree
            # Compare the height and if the current generation has a lower height, move the pointer
            if generated_sentence in all_sentences:
                previous_index = all_sentences.index(generated_sentence)
                previous_height = parent_child_score_map[previous_index][4]
                if curr_node[3] < previous_height:
                    index = previous_index
                else:
                    continue
            else:
                all_sentences.append(generated_sentence)
                index = len(all_sentences) - 1
                top_scores, top_indices = assign_top_scores(top_scores, top_indices, score, index, best_programs)

            parent_child_score_map[index] = (curr_node[1], -1, score, "paraphrase", curr_node[3])
        else:
            assert False, "Operation not supported"

        # Update the indices used in the subtree of the current node. They will not be used further.
        new_child_index_list = curr_node[5].copy()
        new_child_index_list.append(curr_node[1])
        if curr_node[2] != -1:
            new_child_index_list.append(curr_node[2])

        # Update intermediate generation to document sents in the subtree
        document_sents_in_subtree = []
        if curr_node[1] in intermediate_to_document_sents_map:
            document_sents_in_subtree.extend(intermediate_to_document_sents_map[curr_node[1]])
        else:
            document_sents_in_subtree.append(curr_node[1])

        if curr_node[2] != -1:
            if curr_node[2] in intermediate_to_document_sents_map:
                document_sents_in_subtree.extend(intermediate_to_document_sents_map[curr_node[2]])
            else:
                document_sents_in_subtree.append(curr_node[2])

        intermediate_to_document_sents_map[index] = document_sents_in_subtree

        # Add the new generated sentence, combined with all operations to the queue
        # If a sentence has been compressed, don't compress it again
        if curr_node[4] == "compression" or curr_node[4] == "fusion":
            new_height = parent_child_score_map[index][4] + 1
            queue.append((score, index, -1, new_height, "paraphrase", new_child_index_list))

        # If a sentence has been paraphrased, don't paraphrase it again
        if curr_node[4] == "paraphrase" or curr_node[4] == "fusion":
            new_height = parent_child_score_map[index][4] + 1
            queue.append((score, index, -1, new_height, "compression", new_child_index_list))

        # Fuse with indices that haven't been used in the subtree
        # Also, do not fuse with intermediate generations that are created from the same initial sentences
        for temp_index in oracle_sent_indices:
            if temp_index not in new_child_index_list:
                heap_score = max(score, parent_child_score_map[temp_index][2])
                new_height = max(parent_child_score_map[index][4], parent_child_score_map[temp_index][4]) + 1
                queue.append((heap_score, temp_index, index, new_height, "fusion", new_child_index_list))

        for temp_index in range(len(document_sents), len(all_sentences)):
            if temp_index != index and temp_index not in new_child_index_list \
                    and set(intermediate_to_document_sents_map[temp_index]) != set(
                intermediate_to_document_sents_map[index]):
                heap_score = max(score, parent_child_score_map[temp_index][2])
                new_height = max(parent_child_score_map[index][4], parent_child_score_map[temp_index][4]) + 1
                queue.append((heap_score, temp_index, index, new_height, "fusion", new_child_index_list))

    top_trees = construct_tree(parent_child_score_map, top_indices, all_sentences)
    top_tree_strings_with_intermediates, top_tree_strings_without_intermediates = [], []
    for tree in top_trees:
        tree_string_with_intermediates = tree.construct_execution_string(tree.root)
        tree_string_with_intermediates = tree_string_with_intermediates[0] + tree_string_with_intermediates[1]

        intermediate_sentences = tree_string_with_intermediates.split(" | ")
        mapping = {}
        # When saving the strings with intermediate generations, start the intermediate indices from 1.
        for (i, intermediate_sentence) in enumerate(intermediate_sentences):
            if i == 0:
                continue
            parts = intermediate_sentence.split(" --> ")
            assert len(parts) == 2
            mapping[parts[0].strip()] = f"<I{i}>"

        for (key, val) in mapping.items():
            tree_string_with_intermediates = tree_string_with_intermediates.replace(key, val)

        tree_string_without_intermediates = tree.construct_execution_string(tree.root, with_intermediates=False)
        top_tree_strings_with_intermediates.append(tree_string_with_intermediates)
        top_tree_strings_without_intermediates.append(tree_string_without_intermediates[0])

    return top_trees, top_tree_strings_with_intermediates, top_tree_strings_without_intermediates


def search_for_sp(args, document_sents, summary_sents, oracle_sent_indices, compression_model,
                  compression_tokenizer, fusion_model, fusion_tokenizer, paraphrase_model,
                  paraphrase_tokenizer):
    top_programs = [[] for _ in range(args.best_programs)]
    top_program_strings_with_intermediates = [""] * args.best_programs
    top_program_strings_without_intermediates = [""] * args.best_programs
    top_generated_summaries = [""] * args.best_programs
    saved_operations = {}
    for summary_sent in summary_sents:
        top_trees, top_tree_strings_with_intermediates, top_tree_strings_without_intermediates = \
            search_for_tree(args,
                            document_sents,
                            summary_sent,
                            oracle_sent_indices,
                            compression_model,
                            compression_tokenizer,
                            fusion_model,
                            fusion_tokenizer,
                            paraphrase_model,
                            paraphrase_tokenizer,
                            saved_operations)

        for i, (top_tree, top_tree_string_with_intermediates, top_tree_string_without_intermediates) \
                in enumerate(zip(top_trees, top_tree_strings_with_intermediates, top_tree_strings_without_intermediates)):
            top_programs[i].append(top_tree)
            top_program_strings_with_intermediates[i] += "[ " + top_tree_string_with_intermediates + " ] "
            top_program_strings_without_intermediates[i] += "[ " + top_tree_string_without_intermediates + " ] "
            top_generated_summaries[i] += top_tree.root.sentence.strip() + " "

    top_program_strings_with_intermediates = [temp_string.strip() for temp_string in top_program_strings_with_intermediates]
    top_program_strings_without_intermediates = [temp_string.strip() for temp_string in top_program_strings_without_intermediates]
    top_generated_summaries = [temp_string.strip() for temp_string in top_generated_summaries]

    return top_programs, top_program_strings_with_intermediates, top_program_strings_without_intermediates, top_generated_summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='../data/cnndm_dev_sample.tsv', type=str, help='Path to the input file cnndm file')
    parser.add_argument('--output_file', default='../output/searched_sp.tsv', type=str, help='Path to the output file where the SPs will be saved')
    parser.add_argument('--sp_dir', default='../output/searched_sp', type=str, help='Path to the output directory where all the SPs will be rendered')
    parser.add_argument('--best_programs', default=1, type=int, help='Number of top programs to generate per sample')
    parser.add_argument('--start_index', default=0, type=int, help='Start index of the samples to be processed')
    parser.add_argument('--end_index', default=10, type=int, help='End index of the samples to be processed')
    parser.add_argument('--top_k', default=4, type=int, help='Number of top document sentences to consider for SP-Search')
    parser.add_argument('--max_queue_size', default=20, type=int, help='Maximum queue size')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for module execution')
    parser.add_argument('--score_metric', default='rougeL', type=str, help='Metric to optimize for SP-Search')
    parser.add_argument('--paraphrase_model_path', default='../modules/paraphrase', type=str, help='Path to pre-trained Paraphrase Module')
    parser.add_argument('--compression_model_path', default='../modules/compression', type=str, help='Path to pre-trained Compression Module')
    parser.add_argument('--fusion_model_path', default='../modules/fusion', type=str, help='Path to pre-trained fusion Module')
    parser.add_argument('--num_return_sequences', default=5, type=int, help='Number of outputs from each module')
    parser.add_argument('--num_beams', default=5, type=int, help='Beam size for inference from each module')
    parser.add_argument('--max_src_length', default=60, type=int, help='Source length per module')
    parser.add_argument('--max_target_length', default=60, type=int, help='Target length per module')
    parser.add_argument('--temperature', default=1.5, type=float, help='Temperature for inference')
    parser.add_argument('--max_height', default=2, type=int, help='Maximum height of the trees')
    parser.add_argument('--cache_dir', default='../cache', type=str, help='Path to cache directory')
    parser.add_argument('--device', default=0, type=int, help='GPU device')

    args = parser.parse_args()

    torch_device_fusion = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    torch_device_compression = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    torch_device_paraphrase = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    output = open(args.output_file, "a+", encoding="utf-8-sig")
    compression_model, compression_tokenizer = load_compression_model(args.compression_model_path,
                                                                      args.cache_dir,
                                                                      torch_device_compression)
    fusion_model, fusion_tokenizer = load_fusion_model(args.fusion_model_path,
                                                       args.cache_dir,
                                                       torch_device_fusion)
    paraphrase_model, paraphrase_tokenizer = load_paraphrase_model(args.paraphrase_model_path,
                                                                   args.cache_dir,
                                                                   torch_device_paraphrase)

    average_search_time = 0.
    gold_summaries, pred_summaries = [], []
    input_samples = open(args.input_file, "r", encoding="utf-8-sig").read().splitlines()

    for i, input_sample in enumerate(input_samples):
        if i < args.start_index or i > args.end_index:
            continue
        if os.path.exists(os.path.join(args.sp_dir, f"SP_{i}_top1.pdf")):
            continue
        input_parts = input_sample.split("\t")
        if len(input_parts) != 3:
            continue
        document_sents = input_parts[0].split("|||")
        if len(document_sents) == 0:
            continue

        summary = input_parts[1]
        gold_summaries.append(summary.strip())
        summary_sents = nltk.sent_tokenize(summary)

        unigram_scores = np.array([float(unigram_score) for unigram_score in input_parts[2].split(",")])
        top_k = min(args.top_k, len(unigram_scores))
        indices = np.argpartition(unigram_scores, -top_k)[-top_k:].tolist()

        print(f'Processing Sample = {i}')
        start_time = timeit.default_timer()
        try:
            top_programs, top_sp_strings_with_intermediates, top_sp_strings_without_intermediates, \
            top_generated_summaries = search_for_sp(args,
                                                    document_sents,
                                                    summary_sents,
                                                    indices,
                                                    compression_model,
                                                    compression_tokenizer,
                                                    fusion_model,
                                                    fusion_tokenizer,
                                                    paraphrase_model,
                                                    paraphrase_tokenizer)
            output.write(f"{i}\t{summary}\t")
        except Exception:
            print("Some problem with program search!!!")
            continue

        for j, (program, sp_string_with_intermediates, sp_string_without_intermediates, generated_summary) \
                in enumerate(zip(top_programs, top_sp_strings_with_intermediates, top_sp_strings_without_intermediates,
                                 top_generated_summaries)):
            networkx_graph = draw_sp(program)
            try:
                pdf_file_path = os.path.join(args.sp_dir, f"sp_{i}_top{j + 1}")
                networkx_graph.render(pdf_file_path, view=False)
                if os.path.exists(pdf_file_path):
                    os.remove(pdf_file_path)
            except Exception:
                print("Some problem with pdf rendering!!!")
                pass

            rouge_score = compute_rouge([summary], [generated_summary])[args.score_metric]
            pred_summaries.append(generated_summary)

            # Save the identified summaries, the sp strings (with and without intermediate generations) and the rouge score
            output.write(f"{generated_summary}\t{sp_string_with_intermediates}\t{sp_string_without_intermediates}\t{rouge_score}\t")

        end_time = timeit.default_timer()
        time = end_time - start_time
        average_search_time += time
        output.write(f"{time}\n")

        output.flush()
        os.fsync(output.fileno())
