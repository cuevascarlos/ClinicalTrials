
#Import required packages
from transformers import AutoTokenizer, LongformerTokenizerFast, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, PreTrainedTokenizerFast
from datasets import load_dataset, load_metric, load_from_disk
import evaluate
import argparse
import torch
from datasets import ClassLabel, Sequence
import random
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import optuna
import os
import matplotlib.pyplot as plt

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))

def tokenize_and_align_labels(examples):
    #tokenized_inputs = tokenizer(examples["tokens"], truncation=True, max_length=512, is_split_into_words=True)
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
   
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = {
          'accuracy': accuracy_score(true_labels, true_predictions),
          'f1': f1_score(true_labels, true_predictions),
          'classification_report': classification_report(true_labels, true_predictions, output_dict=True)         
    }
    return results

def compute_objective(predictions, labels):
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return f1_score(true_labels, true_predictions)

def generate_csv_comparison(path_data):
    #Read the data
    files = [file for file in os.listdir(path_data) if file.startswith("test_report")]
    print(files)
    
    f1_scores = pd.DataFrame()
    for file in files:
        data = pd.read_csv(f"{path_data}/{file}", delimiter=',')
        f1_scores[file] = data['f1-score'].round(3)

    mean = f1_scores.mean(axis=1).round(3)
    sd = f1_scores.std(axis=1).round(3)
    max = f1_scores.max(axis=1).round(3)
    
    f1_scores['Support'] = data['support']
    f1_scores['Entity'] = data['Unnamed: 0']
    f1_scores.index = f1_scores['Entity']
    f1_scores = f1_scores.drop(columns=['Entity'])
    
    f1_scores['Mean'] = list(mean.astype(str) + ' (+-' + sd.astype(str) + ')')
    f1_scores['Just mean'] = list(mean)
    f1_scores['Max'] = list(max)
    
    #Put Support column the first one
    f1_scores = f1_scores[['Support'] + [col for col in f1_scores.columns if col != 'Support']]

    #Sort the index of the rows
    f1_scores = f1_scores.reindex(['total-participants', 'intervention-participants','control-participants', 'age', 'eligibility', 'ethinicity', 'condition', 'location', 'intervention', 'control', 'outcome', 'outcome-Measure', 'iv-bin-abs', 'cv-bin-abs', 'iv-bin-percent', 'cv-bin-percent', 'iv-cont-mean', 'cv-cont-mean', 'iv-cont-median', 'cv-cont-median', 'iv-cont-sd', 'cv-cont-sd', 'iv-cont-q1', 'cv-cont-q1', 'iv-cont-q3', 'cv-cont-q3', 'micro avg', 'macro avg', 'weighted avg'])

    #Add manually the values of the paper
    biobert = [0.94, 0.85, 0.88, 0.8, 0.74, 0.88, 0.8, 0.76, 0.84, 0.76, 0.81, 0.84, 0.8, 0.82, 0.87, 0.88, 0.81, 0.86, 0.75, 0.79, 0.83, 0.82, 0, 0, 0, 0, None, None, None]
    longformer = [0.95, 0.85, 0.88, 0.87, 0.88, 0.79, 0.79, 0.87, 0.84, 0.81, 0.85, 0.9, 0.82, 0.82, 0.86, 0.85, 0.84, 0.86, 0.69, 0.73, 0.89, 0.89, 0, 0, 0, 0, None, None, None]
    
    f1_scores['BioBERT_paper'] = biobert
    f1_scores['Mean diff BioBERT'] = (f1_scores['Just mean'] - f1_scores['BioBERT_paper']).round(3)
    f1_scores['Max diff BioBERT'] = (f1_scores['Max'] - f1_scores['BioBERT_paper']).round(3)

    f1_scores['Longformer_paper'] = longformer
    f1_scores['Mean diff Longformer'] = (f1_scores['Just mean'] - f1_scores['Longformer_paper']).round(3)
    f1_scores['Max diff Longformer'] = (f1_scores['Max'] - f1_scores['Longformer_paper']).round(3)

    f1_scores = f1_scores.drop(columns=['Just mean'])

    print("Saving file...")
    f1_scores.to_csv(f"{path_data}/f1_scores.csv", sep=',')

def generate_learning_curves(path_data):
    #Read the data
    files = [file for file in os.listdir(path_data) if file.startswith("log_history")]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(files)
    
    eval_f1 = pd.DataFrame()
    for i, file in enumerate(files):
        plt.figure()
        data = pd.read_csv(f"{path_data}/{file}", delimiter=',')

        #Get the entries of the loss column that are not NaN
        loss = data['loss'].dropna().reset_index(drop=True)
        val_loss = data['eval_loss'].dropna().reset_index(drop=True)

        #Get the entries of the eval_f1 column that are not NaN
        eval_f1[f"Training {i}"] = data['eval_f1'].dropna().reset_index(drop=True)

        #Plot the loss and val_loss
        plt.plot(loss, label=f'Train loss')
        plt.plot(val_loss, label=f'Val loss')
        plt.legend()
        plt.title(f"Loss on training and validation set over epochs training {i}")
        #Save the plot
        plt.savefig(f"{path_data}/loss_{i}.png")

    #Plot the eval_f1
    eval_f1.plot()
    plt.legend()
    plt.title("F1-score on validation set over epochs")
    plt.savefig(f"{path_data}/eval_f1.png")

BATCH_SIZE = 8

if __name__ == '__main__':
    #Define the argument parser
    parser = argparse.ArgumentParser(description='Fine-tune a model on NER task')
    parser.add_argument('-d','--data', help='Dataset to work with', required=True)
    parser.add_argument('-m','--model', help='Model checkpoint', required=True)
    parser.add_argument('-s','--save_path', help='Path to save the model', required=False)
    parser.add_argument('-t','--task', help='Task to perform', required=False)

    #Parse the arguments
    args = vars(parser.parse_args())
    model_checkpoint = args['model']

    if args['task'] != None:
        task = args['task']
    else:
        task = 'ner'

    model_name = model_checkpoint.split("/")[-1]
    data_name = args['data'].split("/")[-1]

    if args['save_path'] != None:
        save_path = f"{args['save_path']}/{model_name}-finetuned-{task}"
    else:
        save_path = f"./{model_name}-finetuned-{task}"

    #Load the dataset
    print("LOADING DATASET...")
    data_files = {'train':f"train-00000-of-00001.parquet", 'valid': f"valid-00000-of-00001.parquet", 'test': f"test-00000-of-00001.parquet"}
    datasets = load_dataset('parquet', data_dir = args['data'], data_files=data_files)
    print(datasets)

    label_list = datasets['train'].features[f"{task}_tags"].feature.names
    idsxlabel = {i: label for i, label in enumerate(label_list)}
    labelxids = {label: i for i, label in enumerate(label_list)}
    
    #Preprocessing the data
    #Define the tokenizer
    if model_checkpoint == '../models/longformer-base-4096':
         tokenizer = LongformerTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
    else:
         tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    label_all_tokens = True
    print("TOKENIZING...")
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    #Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    #Hyperparameter fine-tuning
    model_name = model_checkpoint.split("/")[-1]
    print("FINE-TUNING HYPERPARAMETERS...")
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
 
    def objective(trial: optuna.Trial):
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), label2id=labelxids, id2label=idsxlabel)
        print(model)   
        model.to(device)

        args = TrainingArguments(
        save_path+"/HyperparametersSearch",
        evaluation_strategy = "epoch",
        learning_rate=trial.suggest_float("learning_rate", low=2e-5, high=5e-5, log=True),
        weight_decay=trial.suggest_float("weight_decay", 4e-5, 0.01, log=True),
        per_device_train_batch_size= trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        per_device_eval_batch_size= trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32]),
        num_train_epochs=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        greater_is_better = False,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        )
	
        trainer.train()
        #We want to maximize the f1-score in validation set
        predictions, labels, metrics = trainer.predict(tokenized_datasets['valid'])  
        print(f"Validation set metrics: \n {metrics}")
        f1_value = compute_objective(predictions, labels)
        print(f"F1-Score: {f1_value}")
        return f1_value
        

    # We want to maximize the f1-score in validation set
    study = optuna.create_study(study_name="hyper-parameter-search", direction="maximize")
    study.optimize(func=objective, n_trials=15)
    print(f"Best F1-score: {study.best_value}")
    print(f"Best parameters: {study.best_params}")
    print(f"Best trial: {study.best_trial}")
    
  
    #TRAINING
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    for i in range(5):

        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), label2id=labelxids, id2label=idsxlabel)
   
        model.to(device)

        model_name = model_checkpoint.split("/")[-1]
        args = TrainingArguments(
            f"{save_path}/Training_{i}",
            evaluation_strategy = "epoch",
            learning_rate=float(study.best_params['learning_rate']),
            per_device_train_batch_size=int(study.best_params['per_device_train_batch_size']),
            per_device_eval_batch_size=int(study.best_params['per_device_eval_batch_size']),
            weight_decay = float(study.best_params['weight_decay']),
            num_train_epochs=10,
            save_strategy="epoch",
            logging_strategy="epoch",
	        metric_for_best_model="f1",
            load_best_model_at_end=True,
	        save_total_limit=1,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["valid"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        #Train the model
        print("TRAINING...")
   
        print(f"Training iteration {i+1}")
        outputs_train = trainer.train()
        print(outputs_train)
        torch.save(model, f"{save_path}/model_{i}.pt")
        pd.DataFrame(trainer.state.log_history).to_csv(f"{save_path}/log_history_{i}.csv")

        #Evaluate the model
        print("EVALUATING...")
        outputs_test = trainer.evaluate(tokenized_datasets["test"])['eval_classification_report']
        print(outputs_test)
        pd.DataFrame(outputs_test).transpose().to_csv(f"{save_path}/test_report_{i}.csv")

    generate_csv_comparison(save_path)
    generate_learning_curves(save_path)

    
    
