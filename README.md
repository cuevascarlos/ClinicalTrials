# ClinicalTrials
TER (travail d’étude et de recherche) at LISN to extract information from publications of clinical trials.

**Author:** Carlos Cuevas Villarmin

Under the supervision of the professors *Nona Naderi* and *Sarah Cohen-Boulakia*.

*Keywords:* Name Entity Recognition, NER, clinical trials, PICO corpus, Reproducibility.

### Introduction

The main objective of this work is to reproduce the work done in the article ([PICO Corpus: A Publicly Available Corpus to Support Automatic Data Extraction from Biomedical Literature](https://aclanthology.org/2022.wiesp-1.4.pdf)).

The pipeline of the project is tha following:

1. Analysis of the dataset.
2. Preprocessing and creation of datasets understandable by the models.
3. Fine-tune BERT-like models for NER task.

### Analysis of the dataset
The data has been downloaded from this [GitHub repository](https://github.com/sociocom/PICO-Corpus). Each abstract has two associated files:

1. *.txt files:* Files with the abstract plain text.
2. *.ann files:* Annotated files in BRAT format.

The first step is to transform the *.ann files* into *.conll files* in order to have the tagged information in IOB format. There are a lot of tools on internet that make this transformation but finally we have chosen the implementation from [nlplab repository](https://github.com/nlplab/brat). In particular we have used the script `anntoconll.py` and we have modify the function `eliminate_overlaps` to optimize the computation. The used one is the following:

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

It can be seen that T2 and T26 are overlapped and after check which is the correct one we can conclude that T26 is the wrong one with the wrong beginning and ending values. To avoid not considering either of the two we have removed manually T26.

This change is the consequence of the frequency of the total-participants entity being a unit smaller than that reported by the authors. Apparently, they consider it twice.

Once the *.conll files* are read in order to reproduce the corpus statistics of Table 1 in the [paper](https://aclanthology.org/2022.wiesp-1.4.pdf) as the entities are divided into B-ner_tag and I-ner_tag the amount of times each entity appears can be reported considering only the times B-ner_tag appears for each ner_tag. The corpus statistics obtained in our work are the following:

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

**Note.** The p-values reported in the paper were computed based on the strict mode reports using `seqeval` library instead of the default mode. Then, the p-values for each of the options between the F1-scores obtained with exact hyperparameters and rounded hyperparameters for BioBERT models are:

|Entity                     |strict |default|lenient|
|---------------------------|-------|-------|-------|
|total-participants	        |0.6015 |0.8345 |0.9168 |
|intervention-participants  |0.3472 |0.4647 |0.9168 |
|control-participants       |0.4647 |0.3472 |0.7540 |
|age	                    |0.2506 |0.4647 |0.6015 |
|eligibility	            |0.7540 |0.7540 |0.5309 |
|ethinicity                 |0.9168 |1.0000 |0.6761 |
|condition                  |0.6015 |0.6015 |0.7540 |
|location	                |0.7540 |1.0000 |1.0000 |
|intervention	            |0.2506 |0.3472 |0.4647 |
|control	                |0.0758 |0.0758 |0.7540 |
|outcome	                |0.2506 |0.0947 |0.7540 |
|outcome-Measure	        |0.8345 |0.7540 |0.7540 |
|iv-bin-abs	                |0.6015 |0.9168 |0.6015 |
|cv-bin-abs	                |0.2963 |0.8345 |0.4034 |
|iv-bin-percent	            |0.1172 |0.1172 |0.2506 |
|cv-bin-percent	            |0.0947 |0.1745 |0.6015 |
|iv-cont-mean	            |0.4647 |0.1437 |0.4034 |
|cv-cont-mean	            |0.4647 |0.4034 |0.7540 |
|iv-cont-median	            |0.7540 |0.9168 |0.6015 |
|cv-cont-median             |0.4034 |0.5309 |0.4034 |
|iv-cont-sd	                |0.5309 |0.7540 |0.9168 |
|cv-cont-sd	                |0.3472 |0.7540 |0.1745 |
|iv-cont-q1	                |nan    |nan    |nan    |
|cv-cont-q1	                |nan    |nan    |nan    |
|iv-cont-q3	                |nan	|nan    |nan    |
|cv-cont-q3	                |nan	|nan    |nan    |
|micro avg                  |0.6015 |0.4034 |0.4034 |
|macro avg                  |0.3472 |0.4647 |0.4647 |
|weighted avg               |0.3472 |0.3472 |0.9168 |

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

    The script is thought to read directories and files in the hierarchy they were saved with `TrainingNER.py`. For each of the experiments we have defined a folder where all the models trained under each of the splits were saved inside it. If `model` parameter is set to *biobert-base-cased-v1.2* only the reports saved in directories that start with this name will be compared, i.e., the BioBERT models will be analyzed. On the other hand, if `model` is *longformer-base-4096*, LongFomer models reports will be read and compared.

    The script generates a .txt file where the p-values for all the entities, micro, macro and weighted F1-scores are computed and it is saved in the `directory` introduced as input. To compute the significance test for each of the F1-scores (default/strict/lenient) modes the `name` argument of `find_files_with_name` function should be changed to *f1_scores_default_token_level*, *f1_scores_strict_token_level* or *f1_scores_sk_token_level* respectively. By default it is set to strict mode since those values ​​are those reported in the paper.

    Example (Hypothetical scenario): We have run the TrainingNER.py script with split 80-10-10 several times to obtain different hyperparameters and we want know if there is statistically significant difference. Each of the executions are saved under the name *{model_name}/{i}attempt* where $i=1,2,3,...$ and model_name is biobert-base-cased-v1.2 and longformer-base-4096. If we want to compare only BioBERT models the following command will work.
    ```
    python StatisticalAnalysis.py -d "./outputs/Experiment2" -m "biobert-base-cased-v1.2"
    ```


**REMARK.** LongFormer results are not exactly reproducible between runs and have been previously reported by other users; see [issue](https://github.com/huggingface/transformers/issues/12482).  On the other hand, the last two versions of TrainingNER.py ensure the reproducibility of values by fixing the seeds. However, while the reported results of Experiment 1 with BioBERT were executed with these versions, the reported results of Experiment 2 with BioBERT are not exactly reproducible. After the last checks, the outputs of the model trained in the first iteration of the loop is not reproducible due to the use of a previous version of TrainingNER.py where the first iteration was not reproducible. Nevertheless, the results of the other iterations are well reproduced between versions of TrainingNER.py, so the mean and standard deviation outputs will not have a statistically significant difference. Exact reproducibility of the setup, therefore, can be guaranteed by computing Experiment 1 with BioBERT or running by the user Experiment 2 with BioBERT several times under the same GPU and hyperparameters. When the values are not exactly the same (caused by the use of a different GPU or different hyperparameters, among other factors), we encourage the user to conduct a statistical analysis between attempts.

