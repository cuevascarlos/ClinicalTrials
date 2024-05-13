# ClinicalTrials
TER (travail d’étude et de recherche) at LISN to extract information from publications of clinical trials.

**Author:** Carlos Cuevas Villarmin

Under the supervision of the professors *Nona Naderi* and *Sarah Cohen-Boulakia*.

*Keywords:* Name Entity Recognition, NER, clinical trials, PICO corpus.

### Introduction

The main objective of this work is to replicate the work done in the article ([PICO Corpus: A Publicly Available Corpus to Support Automatic Data Extraction from Biomedical Literature](https://aclanthology.org/2022.wiesp-1.4.pdf)).

The pipeline of the project is tha following:

1. Analysis of the dataset.
2. Preprocessing and creation of datasets understandable by the models.
3. Fine-tune BERT-like models for NER task.

### Analysis of the dataset
The data has been downloaded from this [GitHub repository](https://github.com/sociocom/PICO-Corpus). Each abstract has two associated files:

1. *.txt files:* Files with the abstract text.
2. *.ann files:* Annotated files in BRAT format.

The first step is to transform the *.ann files* into *.conll files* in order to have the tagged information in IOB format. There are a lot of functions on internet that make this transformation but finally we have chosen the implementation from [nlplab repository](https://github.com/nlplab/brat). In particular we have used the tool `anntoconll.py` and we have modify the function `eliminate_overlaps` to optimize the computation. The used one is the following:

```
def eliminate_overlaps(textbounds):
    eliminate = {}

    # TODO: avoid O(n^2) overlap check (done?)
    # Create a copy of the list to avoid modifying the original list while
    # added to avoid repeating the same comparison
    textbounds_aux = textbounds[:]

    for t1 in textbounds:
        for t2 in textbounds:
            if t1 is t2: #This should not happen but just in case
                continue
            if t2.start >= t1.end or t2.end <= t1.start:
                continue
            # eliminate shorter
            if t1.end - t1.start > t2.end - t2.start:
                print("Eliminate %s due to overlap with %s" % (
                    t2, t1), file=sys.stderr)
                eliminate[t2] = True
            elif t1.end - t1.start < t2.end - t2.start:
                print("Eliminate %s due to overlap with %s" % (
                    t1, t2), file=sys.stderr)
                eliminate[t1] = True
        # Remove t1 from the list to avoid repeating the same comparison
        textbounds_aux.remove(t1)
        
    return [t for t in textbounds if t not in eliminate]
```

At this point we have a *.conll file* for each text in IOB format which are the files we will work with.

**REMARK:** The file *15023242.ann* contains a mistake that the authors have omitted. 

```
T1	intervention 0 10	Paclitaxel
T2	total-participants 562 565	240
T4	control-participants 673 676	178
T5	eligibility 566 606	patients treated in 6 consecutive trials
T6	intervention-participants 826 828	62
T8	cv-cont-median 1063 1073	148 months
T9	iv-cont-median 1112 1121	45 months
T10	outcome 1156 1198	Estrogen receptor (ER) status was negative
T11	cv-bin-abs 1202 1204	58
T12	cv-bin-percent 1212 1215	33%
T13	iv-bin-abs 1232 1234	40
T14	iv-bin-percent 1242 1245	65%
T15	outcome 1321 1368	objective response rates (complete and partial)
T16	cv-bin-percent 1392 1395	74%
T17	iv-bin-percent 1406 1409	82%
T18	outcome 1416 1444	median overall survival (OS)
T19	outcome 1449 1480	progression-free survival (PFS)
T20	outcome 1631 1640	median OS
T21	cv-cont-median 1651 1660	32 months
T22	iv-cont-median 1671 1680	54 months
T23	outcome 1692 1702	median PFS
T24	cv-cont-median 1713 1722	18 months
T25	iv-cont-median 1733 1742	27 months
T26	total-participants 561 564	240
T3	outcome 1024 1050	median follow-up durations
T7	control 747 796	FAC (5-fluorouracil/doxorubicin/cyclophosphamide)
```

It can be seen that T2 and T26 are overlapped and after check which is the correct one we can conclude that T26 is the wrong one with the wrong beginning and ending values. It has been modified manually to have the same beginning and ending values, it could also have been omitted directly.

This change is the consequence of the frequency of the total-participants entity being a unit smaller than that reported by the authors.

Once the *.conll files* are read in order to reproduce the corpus statistics of Table 1 in the [paper](https://aclanthology.org/2022.wiesp-1.4.pdf) as the entities are divided into B-ner_tag and I-ner_tag the amount of times each entity appears can be reported considering only the times B-ner_tag appears for each ner_tag. The corpus statistics obtained in our work is the following:

|Entity                     |Count|n_files|
|---------------------------|-----|-------|
|B-total-participants	    |1093 |847    |
|B-intervention-participants|887  |674    |
|B-control-participants     |784  |647    |
|B-age	                    |231  |210    |
|B-eligibility	            |925  |864    |
|B-ethinicity               |101  |83     |
|B-condition                |327  |321    |
|B-location	                |186  |168    |
|B-intervention	            |1067 |1011   |
|B-control	                |979  |949    |
|B-outcome	                |5038 |978    |
|B-outcome-Measure	        |1077 |413    |
|B-iv-bin-abs	            |556  |288    |
|B-cv-bin-abs	            |465  |258    |
|B-iv-bin-percent	        |1376 |561    |
|B-cv-bin-percent	        |1148 |520    |
|B-iv-cont-mean	            |1366 |154    |
|B-cv-cont-mean	            |327  |154    |
|B-iv-cont-median	        |270  |140    |
|B-cv-cont-median           |247  |133    |
|B-iv-cont-sd	            |129  |69     |
|B-cv-cont-sd	            |124  |67     |
|B-iv-cont-q1	            |4    |3      |
|B-cv-cont-q1	            |4    |3      |
|B-iv-cont-q3	            |4	  |3      |
|B-cv-cont-q3	            |4	  |3      |

It can be seen that the frequency of total-participants is 1 less than in the paper but also there is a mismatch in the entities outcome and outcome-Measure which appears 15 and 4 times less than in the article. In the file `Preprocessing.ipynb` is the analysis of this mismatch but we can conclude that it is caused by using a different transformation from *.ann files* to *.conll files*. The chosen implementation merges two consecutive words tagged under the same entity so there is only one B-ner_tag instead of two different B-ner_tag. Here is an example:

The *.ann file* is:
```
T8	outcome-Measure 807 903	growth factor [serum hepatocyte growth factor (HGF), vascular endothelial growth factor (VEGF)],
T9	outcome-Measure 904 944	lipid (serum cholesterol, triglycerides)
T10	outcome-Measure 946 962	oxidative damage
```

A part of *.conll file* associated is:

```
396531  24646362  I-outcome-Measure    888  894         factor        11892
396532  24646362  I-outcome-Measure    895  896              (        11892
396533  24646362  I-outcome-Measure    896  900           VEGF        11892
396534  24646362  I-outcome-Measure    900  901              )        11892
396535  24646362  I-outcome-Measure    901  902              ]        11892
396536  24646362  I-outcome-Measure    902  903              ,        11892
396537  24646362  I-outcome-Measure    904  909          lipid        11892
396538  24646362  I-outcome-Measure    910  911              (        11892
396539  24646362  I-outcome-Measure    911  916          serum        11892
396540  24646362  I-outcome-Measure    917  928    cholesterol        11892
396541  24646362  I-outcome-Measure    928  929              ,        11892
396542  24646362  I-outcome-Measure    930  943  triglycerides        11892
396543  24646362  I-outcome-Measure    943  944              )        11892
396545  24646362  B-outcome-Measure    946  955      oxidative        11892
396546  24646362  I-outcome-Measure    956  962         damage        11892
396548  24646362  B-outcome-Measure    967  979   inflammatory        11892
396549  24646362  I-outcome-Measure    980  990     biomarkers        11892
```

It can be seen that T8 and T9 has been merged. All the cases where exist a discrepancy are caused by the same problem. Finding how to fix it in order to obtain the same values as the ones reported in the paper is left at the moment as further work.


### Preprocessing and creation of datasets understandable by the models.

The authors of the paper do not provide precise information about how did they created the dataset for training and testing. The unique related information is that the data was splitted randomly in train set (80%) and test set (20%).

Our problem is in essence a multiclass classification with 26 entities, in BIO format each of them has its corresponding B-ner_tag and I-ner_tag and the class O (outside) is included. At the end, the model will have to learn to classify tokens into 53 entities/classes. 

On the one hand, we have created one dataset without any default split in order to be able to manipulate it later. On the other hand, we create some dataset splits. Firstly, one following the same percentage of abstracts for each set as in the referenced paper 80% for train set and 20% for test set. Secondly, two more splits into train/validation/test sets to follow the standard approach and optimize hyperparameters. In order to do the split keeping a similar representation of all the entities in the sets we have ensured that the frequency of each entity in each set follows a similar distribution as the split (60-20-20 or 70-10-20). The process of the generation of the datasets can be seen in `Preprocessing.ipynb` and the uploaded datasets are in this [HuggingFace repository](https://huggingface.co/datasets/cuevascarlos/PICO-breast-cancer).

### Fine-tune BERT-like models for NER task.

We have chosen the following pretrained models from HuggingFace:

1. [BioBERT](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1)
2. [LongFormer](https://huggingface.co/allenai/longformer-base-4096)

which are the models used in the paper. Additionally, we have consider another model

3. [PubMedBERT](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

Firstly, we optimize the hyperparameters of the models with [Optuna Framework](https://optuna.org/). Concretely, the learning rate, weight decay and batch size (if GPU capacity allows it, otherwise batch size will be 8). Secondly, we train several times the models to avoid bias caused by the random generation of initial weights. Finally, in order to compare the results for each entity we average the f1-scores and we also consider the maximum value. All this process is done in `TrainingNER.py`.

The chosen metrics for analysing the performance of the models is [seqeval library](https://github.com/chakki-works/seqeval) in the strict mode. The models report good results but a deep error analysis has to be taken into account to see the type of errors is done.

## How to use

1. For the data preprocessing it is crucial to clone the repository https://github.com/nlplab/brat.git

2. Data has to be in a folder named "data" where each of the files has its corresponding *.txt* and *.ann* files.

3. For training models with TrainingNER.py:
     
    - Create the environment `conda env create -f environment.yml`
    - Download the processed data and models from HuggingFace.

    The script has several arguments:

    - `mode` option has the possibilities train/test/inference.

        **Train mode:** This mode does hyperparamenter optimization, train several models with different seeds and finally evaluate on test set.

        **Test mode:** Evaluate the models saved locally on test set and compute the metrics at token level and word level (assigning the tag of the first subtoken, there exists the option of considering the most common entity in the subtokens) considering strict and default mode of `seqeval` framework and relaxed/lenient mode using `sklearn` library.
        Finally, a votation is done and the most common prediction with all the models is assigned to the word.

        **Inference mode:** Mode that infers over a given text or text file. It generates a .txt file with the spans that have been identified as entities, along with their start and end in the text, as well as the probability with which they have been classified.

    - `data` directory that contains the .parquet files of the train, development and test set.

    - `model` directory that contains the model that is going to be fine-tuned on NER.

    - `save_path` directory to save all the fine-tuned models, reports, figures and evaluations.

    - `read_file` directory to read the file and infer.

    - `text` raw text to do inference.


