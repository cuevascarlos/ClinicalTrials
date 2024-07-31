# ClinicalTrials
TER (travail d’étude et de recherche) at LISN to extract information from publications of clinical trials.

**Author:** Carlos Cuevas Villarmin

Under the supervision of the professors *Nona Naderi* and *Sarah Cohen-Boulakia*.

*Keywords:* Name Entity Recognition, NER, clinical trials, PICO corpus, Reproducibility.

### Introduction

The main objective of this work is to reproduce the work done in the article ([PICO Corpus: A Publicly Available Corpus to Support Automatic Data Extraction from Biomedical Literature](https://aclanthology.org/2022.wiesp-1.4.pdf)).

The pipeline of the project is the following:

1. Analysis of the dataset.
2. Preprocessing and creation of datasets understandable by the models.
3. Fine-tune BERT-like models for NER task.

### Analysis of the dataset
The data has been downloaded from this [GitHub repository](https://github.com/sociocom/PICO-Corpus). Each abstract has two associated files:

1. *.txt files:* Files with the abstract plain text.
2. *.ann files:* Annotated files in BRAT format.

The first step is to transform the *.ann files* into *.conll files* in order to have the tagged information in IOB format. There are a lot of tools on internet that make this transformation but finally we have chosen the implementation from [nlplab repository](https://github.com/nlplab/brat). In particular we have used the script `anntoconll.py` and we have modify the function `eliminate_overlaps` to optimize the computation in order to not compare two times each pair of entities. The used one is the following:

```
def eliminate_overlaps(textbounds):
    eliminate = {}

    # TODO: avoid O(n^2) overlap check (done?)
    # Create a copy of the list to avoid modifying the original list while
    # added to avoid repeating the same comparison
    textbounds_aux = textbounds[:]

    for t1 in textbounds:
        for t2 in textbounds_aux:
            if t1 is t2: #This should not happen but just in case
                continue
            if t2.start >= t1.end or t2.end <= t1.start:
                continue
            # eliminate shorter
            if t1.end - t1.start >= t2.end - t2.start:
                print("Eliminate %s due to overlap with %s" % (
                    t2, t1), file=sys.stderr)
                eliminate[t2] = True
            else:
                print("Eliminate %s due to overlap with %s" % (
                    t1, t2), file=sys.stderr)
                eliminate[t1] = True
        # Remove t1 from the list to avoid repeating the same comparison
        textbounds_aux.remove(t1)
        
    return [t for t in textbounds if t not in eliminate]
```

At this point we have a *.conll file* for each text in IOB format which are the files we will work with.

**REMARK.** The tool we have used to tranform data into CoNLL with IOB format removes overlapped annotations. The orginal file *15023242.ann* contains one:

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

It can be seen that T2 and T26 are overlapped and the previous function removes T26.

This change is the consequence of the frequency of the total-participants entity being a unit smaller than that reported by the authors. Apparently, they consider it twice.

Apart from the case of the total-participants entity that we have already commented there is a mismatch in the entities outcome and outcome-Measure which appears 15 and 4 times less than in the article. In the file `Preprocessing.ipynb` is the analysis of this mismatch and we can conclude that it is caused by using a different transformation from *.ann files* to *.conll files*. The chosen implementation merges two consecutive chunks tagged under the same entity so there is only one B-ner_tag instead of two different B-ner_tag. Here is an example:

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
```

It can be seen that T8 and T9 has been considered as a single chunk. All the cases where exist a difference are caused by the same problem. So, we can infer that the authors of the dataset consider them separately.


### Preprocessing and creation of datasets understandable by the models.

The authors of the paper do not provide precise information about how did they created the dataset for training and testing. The unique related information is that the data was split randomly in train set (80%) and test set (20%).

Our problem is in essence a multiclass classification with 26 entities, in IOB format each of them has its corresponding B-ner_tag and I-ner_tag and the class O (outside) is included. At the end, the model will have to learn to classify tokens into 53 entities/classes. 

On the one hand, we have created one dataset without any default split in order to be able to manipulate it later. On the other hand, we create some dataset splits. Firstly, one following the same percentage of abstracts for each set as in the referenced paper 80% for train set and 20% for test set, when using this dataset split we name it Experiment 1. Secondly, one split into train/validation/test sets to follow the standard approach and optimize hyperparameters. In order to do the split keeping a similar representation of all the entities in the sets we have ensured that the frequency of each entity in each set follows a similar distribution as the split (80-10-10) which is named Experiment 2. The process of the generation of the datasets can be seen in `Preprocessing.ipynb` and the uploaded datasets are in this [HuggingFace repository](https://huggingface.co/datasets/cuevascarlos/PICO-breast-cancer).

### Fine-tune BERT-like models for NER task.

We have chosen the following pretrained models from HuggingFace:

1. [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
2. [LongFormer](https://huggingface.co/allenai/longformer-base-4096)

which are the models used in the paper. We have download each of them locally in folders under the name `models/biobert-base-cased-v1.2` and `models/longformer-base-4096`. Another hierarchy does not guarantee that the scripts will work.

The hyperparameters used in Experiment 1 can be downloaded from `Hyperparameters/Experiment1.pkl`. For Experiment 2, we optimize the hyperparameters of the models with [Optuna Framework](https://optuna.org/). Concretely, the learning rate, weight decay and batch size. The hyperparameters obtained in the optimization are in `Hyperparameters/Experiment2_biobert.pkl` and `Hyperparameters/Experiment2_longformer.pkl`.

Secondly, with these fixed hyperparameters we train 5 times the models to avoid bias caused by the random generation of initial weights. Finally, in order to compare the results for each entity we average the f1-scores and we also compute the maximum value. All this process is done in `TrainingNER.py`.

The reported metrics for analysing the performance of the models are F1-scores computed using the [seqeval library](https://github.com/chakki-works/seqeval) in the default mode which as the author says it is compatible with the standard CoNLL evaluation. However, the code computes at the same time the strict mode of `seqeval` (more information in the link above) and the relaxed/lenient mode using [sklearn library](https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.classification_report.html).

## How to use

Python version: 3.10.13

1. For the data preprocessing it is crucial to clone the repository https://github.com/nlplab/brat.git and recommended to modify the function explained above.

2. Data has to be in a folder named "data" where each of the files has its corresponding *.txt* and *.ann* files. After preprocessing the *.conll* files will be saved in the same data folder.

3. For training models with `TrainingNER.py`:
     
    - Create the environment `conda env create -f environment.yml`
    - Download the processed data (for example in `./datasets/Experiment*`) and models from HuggingFace (`./models/*`). If models are not save as this it is not ensured that the code will work.

    The script has several arguments:

    - `mode` option has the possibilities train/test/inference.

        **Train mode:** This mode does hyperparamenter optimization, train several models with different seeds and finally evaluate on test set.

        The hyperparameter optimization process is done if: 
            1. There is no .pkl file with pre-defined hyperparameters in the directory where the outputs will be saved or a directory to read a .pkl file is not provided.
            2. The variable `HYPERPARAMETERS_SEARCH` is set to True.

        Otherwise, the hyperparameters used in Experiment 1 are defined by default. They can be modified to suit the user.

        **Test mode:** Evaluate the models saved locally on test set and compute the metrics at token level and word level (assigning the tag of the first subtoken, there exists the option of considering the most common entity in the subtokens) considering strict and default mode of `seqeval` framework and relaxed/lenient mode using `sklearn` library.
        Finally, a votation is done and the most common prediction with all the models is assigned to the word.

        **Inference mode:** Mode that infers over a given text or text file. It generates a .txt file with the spans that have been identified as entities, along with their start and end in the text, as well as the probability with which they have been classified.

    - `model` directory that contains the model that is going to be fine-tuned on NER.

    - `data` directory that contains the .parquet files of the train, validation and test set.

    - `save_path` directory to save all the fine-tuned models, reports, figures and evaluations.

        The script will create a folder inside the directory with the name `model_name-finetuned-ner`.

    - `hyperparam` directory to read the .pkl file with desired hyperparameters.

    - `read_file` directory to read the file and infer.

    - `text` raw text to do inference.

    - `experiment` (Just for reproducibility interests) The number of the experiment it is aimed to be reproduced "1" or "2". It automates some parts of the code that in other case should be changed manually from one experiment to another.
    If `--experiment "1"`, `HYPERPARAMETERS_SEARCH` is set to false and hyperparameter optimization is not done.


    To run each experiment with the same hyperparameters it can be done as:
    ```
    #Experiment 1 - BioBERT
    python TrainingNER.py -mode "train" -d "./datasets/Experiment1" -m "./models/biobert-base-cased-v1.2" -s "./outputs/Experiment1/biobert-base-cased-v1.2-1attempt" -exp "1" -hy "./Hyperparameters/Experiment1.pkl

    #Experiment 1 -LongFormer
    python TrainingNER.py -mode "train" -d "./datasets/Experiment1" -m "./models/longformer-base-4096" -s "./outputs/Experiment1/longformer-base-4096-1attempt" -exp "1" -hy "./Hyperparameters/Experiment1.pkl

    #Experiment 2 - BioBERT
    python TrainingNER.py -mode "train" -d "./datasets/Experiment2" -m "./models/biobert-base-cased-v1.2" -s "./outputs/Experiment2/biobert-base-cased-v1.2-1attempt" -exp "2" -hy "./Hyperparameters/Experiment2_biobert.pkl

    #Experiment 2 - LongFormer 
    python TrainingNER.py -mode "train" -d "./datasets/Experiment2" -m "./models/longformer-base-4096" -s "./outputs/Experiment2/longformer-base-4096-1attempt" -exp "2" -hy "./Hyperparameters/Experiment2_longformer.pkl
    ```
    The script will save everything in directories as for example `./outputs/Experiment2/biobert-base-cased-v1.2-1attempt/biobert-base-cased-v1.2-finetuned-ner`.

    The experiment argument can be removed for experiment 2 and the -hy argument can be deleted in Experiment 1 in case default values have not been modified. If interested in carrying out hyperparameter optimization, the -hy argument have to be omitted for Experiment 2.

    To ensure that the statistical analysis in the next step works, all runs performed in each experiment (i.e., for each data split) must be stored in the same folder. The directory corresponding to each run should start with the model name used, which will serve later as an identifier. To guarantee everything works correctly, it should either be *biobert-base-cased-v1.2* or *longformer-base-4096* (or the name of the version used).

4. Statistical analysis with `StatisticalAnalysis.py`

    The script has two arguments:

    - `directory` Directory to search for files.

    - `model` Model on which the analysis wants to be done.

    - `metric` (sk/default/strict) to compare results between models. Note that sk means the case of lenient mode.

    The script is thought to read directories and files in the hierarchy they were saved with `TrainingNER.py`. For each of the experiments we have defined a folder where all the models trained under each of the splits were saved inside it. If `model` parameter is set to *biobert-base-cased-v1.2* only the reports saved in directories that start with this name will be compared, i.e., the BioBERT models will be analyzed. On the other hand, if `model` is *longformer-base-4096*, LongFomer models reports will be read and compared.

    The script generates a .txt file where the p-values for all the entities, micro, macro and weighted F1-scores are computed and it is saved in the `directory` introduced as input.

    Example (Hypothetical scenario): We have run the TrainingNER.py script with split 80-10-10 several times to obtain different hyperparameters and we want know if there is statistically significant difference. Each of the executions are saved under the name *{model_name}/{i}attempt* where $i=1,2,3,...$ and model_name is biobert-base-cased-v1.2 and longformer-base-4096. If we want to compare only BioBERT models in default mode the following command will work.
    ```
    python StatisticalAnalysis.py -d "./outputs/Experiment2" -m "biobert-base-cased-v1.2" --metric "default"
    ```

    The p-values for default mode between the F1-scores obtained with exact hyperparameters and rounded hyperparameters for BioBERT models are:

    | Entity                     | p-value |
    |----------------------------|---------|
    | total-participants         | 0.6761  |
    | intervention-participants  | 0.4647  |
    | control-participants       | 0.3472  |
    | age                        | 0.9168  |
    | eligibility                | 0.7540  |
    | ethinicity                 | 1.0000  |
    | condition                  | 0.9168  |
    | location                   | 1.0000  |
    | intervention               | 0.1745  |
    | control                    | 0.1745  |
    | outcome                    | 0.2101  |
    | outcome-Measure            | 0.9168  |
    | iv-bin-abs                 | 1.0000  |
    | cv-bin-abs                 | 0.7540  |
    | iv-bin-percent             | 0.2506  |
    | cv-bin-percent             | 0.1745  |
    | iv-cont-mean               | 0.2506  |
    | cv-cont-mean               | 0.4034  |
    | iv-cont-median             | 0.9168  |
    | cv-cont-median             | 0.5309  |
    | iv-cont-sd                 | 0.5309  |
    | cv-cont-sd                 | 0.7540  |
    | iv-cont-q1                 | nan     |
    | cv-cont-q1                 | nan     |
    | iv-cont-q3                 | nan     |
    | cv-cont-q3                 | nan     |
    | micro avg                  | 0.8345  |
    | macro avg                  | 0.9168  |
    | weighted avg               | 0.7540  |


**REMARK.** LongFormer results are not exactly reproducible between runs and have been previously reported by other users; see [issue](https://github.com/huggingface/transformers/issues/12482). Exact reproducibility of the setup, therefore, can be guaranteed by computing Experiment 1 or 2 with BioBERT. When the values are not exactly the same (caused by the use of a different GPU or different hyperparameters, among other factors), we encourage the user to conduct a statistical analysis between attempts. 

Here are the results of the experiments using the Longformer model in the case of using default mode with the library seqeval:

| Entity                       | Mutinda et al.         | Experiment 1           | Experiment 2           |
|------------------------------|------------------------|------------------------|------------------------|
| **Participants**             |                        |                        |                        |
| total-participants           | 0.95                   | 0.8971 (+-0.0091)      | 0.9263 (+-0.0119)      |
| intervention-participants    | 0.85                   | 0.7958 (+-0.0206)      | 0.8435 (+-0.0161)      |
| control-participants         | 0.88                   | 0.8468 (+-0.0092)      | 0.8703 (+-0.0159)      |
| age                          | 0.87                   | 0.5824 (+-0.0366)      | 0.6148 (+-0.0415)      |
| eligibility                  | 0.88                   | 0.5780 (+-0.0091)      | 0.5934 (+-0.0435)      |
| ethinicity                   | 0.79                   | 0.5524 (+-0.0458)      | 0.7421 (+-0.0611)      |
| condition                    | 0.79                   | 0.6424 (+-0.0177)      | 0.7579 (+-0.0351)      |
| location                     | 0.87                   | 0.5951 (+-0.0556)      | 0.6595 (+-0.0637)      |
| **Intervention & Control**   |                        |                        |                        |
| intervention                 | 0.84                   | 0.7612 (+-0.0148)      | 0.7794 (+-0.0293)      |
| control                      | 0.81                   | 0.6564 (+-0.0190)      | 0.6386 (+-0.0202)      |
| **Outcomes**                 |                        |                        |                        |
| outcome                      | 0.85                   | 0.6303 (+-0.0131)      | 0.6829 (+-0.0087)      |
| outcome-Measure              | 0.90                   | 0.7352 (+-0.0144)      | 0.7276 (+-0.0207)      |
| iv-bin-abs                   | 0.82                   | 0.7458 (+-0.0292)      | 0.7804 (+-0.0111)      |
| cv-bin-abs                   | 0.82                   | 0.7970 (+-0.0274)      | 0.8246 (+-0.0191)      |
| iv-bin-percent               | 0.86                   | 0.7230 (+-0.0255)      | 0.7314 (+-0.0379)      |
| cv-bin-percent               | 0.85                   | 0.7725 (+-0.0242)      | 0.792 (+-0.0301)       |
| iv-cont-mean                 | 0.84                   | 0.5385 (+-0.0288)      | 0.4421 (+-0.0581)      |
| cv-cont-mean                 | 0.86                   | 0.5121 (+-0.0293)      | 0.3890 (+-0.0199)      |
| iv-cont-median               | 0.69                   | 0.6462 (+-0.0239)      | 0.7838 (+-0.0288)      |
| cv-cont-median               | 0.73                   | 0.6721 (+-0.0196)      | 0.7927 (+-0.0258)      |
| iv-cont-sd                   | 0.89                   | 0.4644 (+-0.1047)      | 0.7117 (+-0.0737)      |
| cv-cont-sd                   | 0.89                   | 0.5377 (+-0.1051)      | 0.7232 (+-0.1472)      |
| iv-cont-q1                   | 0                      | 0.0000 (+-0.0000)      | nan                    |
| cv-cont-q1                   | 0                      | 0.0000 (+-0.0000)      | nan                    |
| iv-cont-q3                   | 0                      | 0.0000 (+-0.0000)      | nan                    |
| cv-cont-q3                   | 0                      | 0.0000 (+-0.0000)      | nan                    |
| micro avg                    | nan                    | 0.6964 (+-0.0064)      | 0.7325 (+-0.0051)      |
| macro avg                    | 0.7127                 | 0.5647 (+-0.0088)      | 0.7185 (+-0.0103)      |
| weighted avg                 | 0.8530                 | 0.6979 (+-0.0061)      | 0.7343 (+-0.0053)      |


On the other hand, the results with the library scikit-learn are:

| Entity                       | Experiment 2           | Mean - Mutinda et al.         |
|------------------------------|------------------------|-------------------------------|
| **Participants**             |                        |                               |
| total-participants           | 0.9552 (+-0.0089)      | 0.0052                        |
| intervention-participants    | 0.8742 (+-0.0121)      | 0.0242                        |
| control-participants         | 0.8977 (+-0.0190)      | 0.0177                        |
| age                          | 0.9317 (+-0.0075)      | 0.0617                        |
| eligibility                  | 0.8616 (+-0.0245)      | -0.0184                       |
| ethinicity                   | 0.6753 (+-0.0609)      | -0.1147                       |
| condition                    | 0.9350 (+-0.0391)      | 0.1450                        |
| location                     | 0.9308 (+-0.0207)      | 0.0608                        |
| **Intervention & Control**   |                        |                               |
| intervention                 | 0.8505 (+-0.0117)      | 0.0105                        |
| control                      | 0.7555 (+-0.0177)      | -0.0545                       |
| **Outcome**                  |                        |                               |
| outcome                      | 0.8809 (+-0.0029)      | 0.0309                        |
| outcome-Measure              | 0.9665 (+-0.0074)      | 0.0665                        |
| iv-bin-abs                   | 0.8787 (+-0.0297)      | 0.0587                        |
| cv-bin-abs                   | 0.9028 (+-0.0249)      | 0.0828                        |
| iv-bin-percent               | 0.8530 (+-0.0207)      | -0.0070                       |
| cv-bin-percent               | 0.8640 (+-0.0179)      | 0.0140                        |
| iv-cont-mean                 | 0.6472 (+-0.0308)      | -0.1928                       |
| cv-cont-mean                 | 0.5498 (+-0.0366)      | -0.3102                       |
| iv-cont-median               | 0.8151 (+-0.0209)      | 0.1251                        |
| cv-cont-median               | 0.8171 (+-0.0108)      | 0.0871                        |
| iv-cont-sd                   | 0.8787 (+-0.0431)      | -0.0113                       |
| cv-cont-sd                   | 0.8625 (+-0.0313)      | -0.0275                       |
| iv-cont-q1                   | nan                    | nan                           |
| cv-cont-q1                   | nan                    | nan                           |
| iv-cont-q3                   | nan                    | nan                           |
| cv-cont-q3                   | nan                    | nan                           |
| accuracy                     | 0.7993 (+-0.0076)      | nan                           |
| macro avg                    | 0.8080 (+-0.0040)      | 0.0953                        |
| weighted avg                 | 0.8705 (+-0.0052)      | 0.0175                        |


