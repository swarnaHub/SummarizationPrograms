# Summarization Programs (ICLR 2023)

[Summarization Programs: Interpretable Abstractive Summarization with Neural Modular Trees](https://arxiv.org/abs/2209.10492)

[Swarnadeep Saha](https://swarnahub.github.io/), [Shiyue Zhang](https://www.cs.unc.edu/~shiyue/), [Peter Hase](https://peterbhase.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

![image](./assets/sum_prog.png)

![image](./assets/rq_overview.png)

## Installation
This repository is tested on Python 3.8.12.  
You should install SummarizationPrograms on a virtual environment. All dependencies can be installed as follows:
```
pip install -r requirements.txt
```

## Dataset
We provide a small sample of the CNN/DM validation set in `data` folder. Each line contains the source document, the gold summary and unigram overlap percentages of each source sentence with respect to the summary. You can also pre-process your own Summarization dataset in the same format for running SP-Search.

For CNN/DM and XSum, we release the original samples and the searched programs (SP-Search) outputs [here](https://drive.google.com/file/d/1O4JQ6iN9l0bien6hl0b55l9IKXlxpROr/view?usp=sharing).

The `documents` folder contains the samples, pre-processed as discussed above. The SP-Search outputs are represented as follows.

Each line is a tab-separated entry consisting of the following:

- Index (according to the original sample ID in `documents` folder)
- Gold Summary (same as the summaries in `documents` folder)
- SP_Search Summary (the searched summary that emulates the gold/human summary)
- SP_Search program with intermediate generations (S1, S2, etc denote document sentences. I1, I2, etc denote intermediate generations after executing a neural module)
- SP_Search program without intermediate generations (a more compact representation of the previous field. each tree is separated by square brackets)
- ROUGE score between gold and SP_Search summary

## RQ1: SP-Search

In order to identify Summarization Programs for human summaries, execute the following steps.
```
cd sp_search
python main.py
```
The pre-trained modules are available for download [here](https://drive.google.com/drive/folders/1Wn9ZHF91hFbYC3cGNnAaWZe-TihF4taI?usp=sharing). For paraphrase, we directly used the model available [here](https://huggingface.co/tuner007/pegasus_paraphrase). Download the other two modules and place them inside the `modules` directory.

Upon running the search, you will see outputs similar to what's there in the `output` folder. The `sp_search.tsv` file will save the Summarization Programs and the corresponding summaries. The folder `sp_search` will save the SPs in individual pdfs for visualization.

Compute ROUGE scores for the SP-Search summaries by running
```
cd scripts
python compute_spsearch_rouge.py
```
## RQ2: SP Generation Models

Generating summaries via SP has three steps.

- First, we fine-tune a BART model leveraging the SP-Search programs. Use the script `scripts/train_sp_gen.sh`. You need to process the training file in json format according to the HuggingFace transformers library.
- Second, do inference from this model using `sp_model/eval_sp.py` to generate intermediate SPs.
- Third, execute these generated programs via the pre-trained modules to obtain the final summaries. The script is `sp_model/execute_sp.py`.

### Citation
```
@inproceedings{saha2023summarization,
  title={Summarization programs: Interpretable abstractive summarization with neural modular trees},
  author={Saha, Swarnadeep and Zhang, Shiyue and Hase, Peter and Bansal, Mohit},
  booktitle={ICLR},
  year={2023}
}
```
