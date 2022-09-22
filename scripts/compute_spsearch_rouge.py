from datasets import load_metric
import nltk

metric = load_metric("rouge", cache_dir="../cache", seed=42)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_rouge(gold_summaries, pred_summaries):
    pred_summaries, gold_summaries = postprocess_text(pred_summaries, gold_summaries)
    result = metric.compute(predictions=pred_summaries, references=gold_summaries, use_stemmer=True)
    return {key: value.mid.fmeasure * 100 for key, value in result.items()}


gold_summaries, spsearch_summaries, exec_times = [], [], []
with open("../output/searched_sp.tsv", "r", encoding="utf-8-sig") as f:
    lines = f.read().splitlines()
    for (i, line) in enumerate(lines):
        parts = line.split("\t")
        gold_summaries.append(parts[1].strip())
        spsearch_summaries.append(parts[2].strip())
        exec_times.append(float(parts[-1].strip()))

    assert len(gold_summaries) == len(spsearch_summaries)
    assert len(exec_times) == len(gold_summaries)

    overall_rouge_scores = compute_rouge(gold_summaries, spsearch_summaries)
    average_exec_time = sum(exec_times) / len(exec_times)
    print(f'Number of samples = {len(gold_summaries)}')
    print(f'Rouge1 Score = {overall_rouge_scores["rouge1"]}')
    print(f'Rouge2 Score = {overall_rouge_scores["rouge2"]}')
    print(f'RougeL Score = {overall_rouge_scores["rougeL"]}')
    print(f'RougeLsum Score = {overall_rouge_scores["rougeLsum"]}')
    print(f'Average Execution Time (in secs) = {average_exec_time}')
