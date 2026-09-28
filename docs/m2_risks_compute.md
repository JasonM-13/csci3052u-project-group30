# M2: Risks, Feasibility & Compute Plan (issue #5)

## Dataset risks

### Risk 1: labels come from the site rating, not from someone reading the review

The dataset was scraped from OpenRice with OpenRice's labels smile, ok, cry. There could be a mismatch between the review and the labels causing the model to misclassify 
labels. One such label is train.jsonl id 3709, label ok. We won't relabel because we're reproducing CantoNLU on the same data.

### Risk 2: the dataset is artificially balanced

The dataset has already been artificially balanced upstream. The original data source mentions that "In OpenRice, the number of "smile" reviews far outnumber the other two. 
This dataset is exactly balanced with 1/3 from each class." (source: toastynews/openrice-senti), so a score on this balanced test set won't tell us how the model
does on real OpenRice traffic, where most reviews are smile. 

### Risk 3: reviews are longer than the model's input window

The median review is 288 characters and the longest is 4720. BERT-base reads at most 512 tokens, and for Chinese one character is roughly one token, so about 12% of reviews get 
cut off. The paper doesn't say what max length it used, so we'll use 512 and note the truncation.

## Class imbalance

Every split is exactly balanced, 3333 each in train and 333 each in valid and test, so always guessing smile scores 33% here but would score much higher on real OpenRice. We report this as a limitation rather than rebalancing. The baseline we are going to use is 33%, which comes from always guessing the same label.

## Where training will run

Training will run on Khalid-T's machine. Specs: Linux, NVIDIA RTX 5060 Ti with 16 GB VRAM, CUDA 13.4. If Khalid's machine isn't able to provide the training needed for the model, then we are going to use a Google Colab cloud GPU to get the training done. BERT-base at batch 16 and length 512 needs roughly 7 GB with mixed precision, so it fits in 16 GB.

## Noisy labels

The longest review is train id 3709 and is labeled ok. The review is mixed: the person is complaining, then says it's good. Some reviews may be mixed, with both good and bad in them making it hard for the model to classify it to one label. Mixed reviews might mess with the model's ability to correctly classify the label. We know about it because I read it. We won't relabel, we'll read 20 random reviews to see how common mixing is, and in error analysis we'll separate model mistakes from label mistakes.

## Sources

data/sentiment/readme.md, eda/eda.ipynb, toastynews/openrice-senti (CC-BY-4.0), CantoNLU paper Tables 2 and 3.
