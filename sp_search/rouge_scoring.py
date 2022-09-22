import nltk
from datasets import load_metric

metric = load_metric("rouge", cache_dir="../cache", seed=42)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_rouge(gold_summaries, pred_summaries, use_agregator=True):
    assert len(gold_summaries) == len(pred_summaries), "Gold and prediction not of same length"
    pred_summaries, gold_summaries = postprocess_text(pred_summaries, gold_summaries)
    result = metric.compute(predictions=pred_summaries, references=gold_summaries, use_stemmer=True,
                            use_agregator=use_agregator)
    if use_agregator:
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}
    else:
        return result
