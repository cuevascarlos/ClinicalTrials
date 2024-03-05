# ClinicalTrials
TER (travail d’étude et de recherche) at LISN to extract information from publications of clinical trials.

**Author:** Carlos Cuevas Villarmin

Under the supervision of the professors *Nona Naderi* and *Sarah Cohen-Boulakia*.

*Keywords:* Name Entity Recognition, NER, clinical trials, PICO corpus.

### Introduction

The first objective of this work is to replicate the work done in the article [PICO Corpus: A Publicly Available Corpus to Support Automatic Data
Extraction from Biomedical Literature]([PICO Corpus: A Publicly Available Corpus to Support Automatic Data](https://aclanthology.org/2022.wiesp-1.4.pdf)).

The pipeline of the project is tha following:

1. Analysis of the dataset.
2. Preprocessing and creation of datasets understandable by the models.
3. Fine-tune BioBERT and LongFormer models with out datasets.

### Analysis of the dataset
The data has been downloaded from this [GitHub repository](https://github.com/sociocom/PICO-Corpus). Each abstract has two associated files:

1. *.txt files:* Files with the abstract text.
2. *.ann files:* Annotated files from in BRAT format.

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

At this point we have a *.conll file* for each text IN IOB format which are the files we will work with.

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



### Further work
