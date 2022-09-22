from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_paraphrase_model(model_path, cache_dir, device):
    print("Loading paraphrase Model")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=cache_dir).to(device)
    print("Done")

    return model, tokenizer


def load_compression_model(model_path, cache_dir, device):
    print("Loading compression Model")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", cache_dir=cache_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    print("Done")

    return model, tokenizer


def load_fusion_model(model_path, cache_dir, device):
    print("Loading fusion Model")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", cache_dir=cache_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    print("Done")

    return model, tokenizer
