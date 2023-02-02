import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def check_validity(predicted_program_string, sent_ids):
    predicted_program_string = predicted_program_string.replace("  ", " ")
    predicted_program_string = predicted_program_string.replace("] (", "] [ (")
    predicted_program_string = predicted_program_string.replace("))", ") )")
    trees = predicted_program_string[1:-1].split("[")
    for tree in trees:
        tree = tree.strip()
        if tree.endswith("]"):
            tree = tree[:-1].strip()
        if tree.count("(") != tree.count(")"):
            return False, "First bracket did not match"
        operation_stack, operator_stack = [], []
        tokens = tree.split(" ")
        for token in tokens:
            token = token.strip()
            if token == "(" or token == "":
                continue
            elif token == "fusion" or token == "compression" or token == "paraphrase":
                operation_stack.append(token)
            elif re.match("^<S([0-9]|[1-9][0-9]|[1-9][0-9][0-9])>$", token):
                if token in sent_ids:
                    operator_stack.append(token)
                else:
                    return False, "Some other sentence ID encountered"
            elif token == ")":
                if len(operation_stack) == 0:
                    continue
                curr_operation = operation_stack.pop()
                # Execute operations with dummy outputs because we only need to check validity here
                if curr_operation == "fusion":
                    if len(operator_stack) < 2:
                        return False, "Problem with fusion"
                    operator2 = operator_stack.pop()
                    operator1 = operator_stack.pop()
                    generation = "fusion" + operator2 + operator1
                    operator_stack.append(generation)
                elif curr_operation == "compression":
                    if len(operator_stack) == 0:
                        return False, "Problem with compression"
                    operator = operator_stack.pop()
                    generation = "compression" + operator
                    operator_stack.append(generation)
                elif curr_operation == "paraphrase":
                    if len(operator_stack) == 0:
                        return False, "Problem with paraphrase"
                    operator = operator_stack.pop()
                    generation = "paraphrase" + operator
                    operator_stack.append(generation)
                else:
                    return False, "Some other operation found in stack"
            else:
                return False, "Some other token encountered"

        if len(operation_stack) != 0:
            return False, "Operation stack not empty"

        if len(operator_stack) != 1:
            return False, "Operators left in stack"

    return True, "Valid SP"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../models/extract-and-build', type=str, help='Path to trained model')
    parser.add_argument('--test_file', default='../data/test.json', type=str, help='Path to the test file')
    parser.add_argument('--cache_dir', default='../cache', type=str)
    parser.add_argument('--test_generations', default='../models/extract-and-build/generations.txt', type=str, help='Path to where the predicted programs will be saved')
    parser.add_argument('--max_src_length', default=512, type=int)
    parser.add_argument('--max_target_length', default=100, type=int)
    parser.add_argument('--num_beams', default=10, type=int)
    parser.add_argument('--num_return_sequences', default=10, type=int)
    parser.add_argument('--temperature', default=0, type=float)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, cache_dir=args.cache_dir).to(device)

    test_lines = open(args.test_file, "r", encoding="utf-8-sig").read().splitlines()
    output = open(args.test_generations, "w", encoding="utf-8-sig")

    matched_count = 0
    for (i, test_line) in enumerate(test_lines):
        input = json.loads(test_line)["text"]
        batch_tokens = tokenizer([input], truncation=True, padding='longest', max_length=args.max_src_length, return_tensors="pt").to(device)
        generated = model.generate(**batch_tokens, max_length=args.max_target_length, num_beams=args.num_beams,
                                   num_return_sequences=args.num_return_sequences, temperature=args.temperature,
                                   return_dict_in_generate=False, output_scores=False, min_length=10, length_penalty=2)
        programs = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True, spaces_between_special_tokens=False)
        is_valid = False

        words = input.split(" ")
        sentence_markers = []
        for word in words:
            if re.match("^<S([0-9]|[1-9][0-9]|[1-9][0-9][0-9])>$", word):
                sentence_markers.append(word)

        for program in programs:
            temp_valid, error_type = check_validity(program, sentence_markers)
            if temp_valid:
                is_valid = True
                output.write(str(i) + "\t" + program + "\tvalid\n")
                matched_count += 1
                break

        if not is_valid:
            temp_valid, error_type = check_validity(programs[0], sentence_markers)
            output.write(str(i) + "\t[ " + " ] [ ".join(sentence_markers) + " ]" + "\t" + error_type + "\n")