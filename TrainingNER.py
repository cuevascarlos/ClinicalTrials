#Import required packages
from transformers import AutoTokenizer, LongformerTokenizerFast, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
import argparse
import torch
from datasets import Dataset
import random
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import optuna
import os
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.metrics import classification_report as classification_report_sk
from collections import Counter

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

def tokenize_and_align_labels(examples):
    '''
    Tokenize the sentences and align the labels with the tokens.

    This function takes a list of sentences and their corresponding labels, tokenizes the sentences, and aligns the labels with the tokens.
    The function handles special tokens by assigning them a label of -100, so they are automatically ignored in the loss function.
    The function also handles the case where a word is split into multiple tokens, assigning the label to the first token and either the same label or -100 to the other tokens, depending on the label_all_tokens flag.

    Args:
        examples (dict): A dictionary containing two keys:
            - "tokens": A list of sentences, where each sentence is a list of words.
            - "{task}_tags": A list of lists, each containing the labels for a sentence.

    Returns:
        tokenized_inputs (dict): A dictionary containing the tokenized sentences and their corresponding labels and word_ids.
            - "input_ids", "attention_mask", "token_type_ids": Lists of tokenized sentences.
            - "word_ids": A list of lists, each containing the word_ids for a sentence.
            - "labels": A list of lists, each containing the aligned labels for a sentence.
    '''
    if flag_tokenizer == 'LongFormer':
        tokenized_inputs = tokenizer(examples["tokens"], padding=True, truncation=True, max_length=1024, is_split_into_words=True)
    else:
        tokenized_inputs = tokenizer(examples["tokens"], padding=True, truncation=True, max_length=512, is_split_into_words=True)
    
    labels = []
    word_ids = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids.append(tokenized_inputs.word_ids(batch_index=i))
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids[-1]:
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
    
    tokenized_inputs['word_ids'] = word_ids
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def generate_true_predictions_and_labels(predictions, labels, label_list, mode = None):
    '''
    Generate true predictions and labels by removing special tokens.

    This function takes predictions and labels and filters out special tokens (with label -100).
    If mode is set to 'sklearn', it also filters out outside tokens (with label 0).
    It returns the true predictions and labels as lists of lists.

    Args:
        predictions (np.array): The predicted probabilities for each label. 
                            This is a 2D array where the first dimension is the number of examples and the second dimension is the number of possible labels.
        labels (list of list of int): The true labels for each example. 
                                  This is a list of lists where each inner list contains the labels for one example.
        label_list (list): A list of labels corresponding to the indices in the predictions.
        mode (str, optional): The mode to use for filtering labels. If set to 'sklearn', outside tokens (with label 0) are also filtered out.

    Returns:
        true_predictions (list of list): A list of lists, each containing the true predictions for a sentence.
        true_labels (list of list): A list of lists, each containing the true labels for a sentence.
    '''


    if mode == 'sklearn':
        # Remove ignored index (special tokens) and outside tokens
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100 and l!=0]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100 and l!=0]
            for prediction, label in zip(predictions, labels)
        ]
    else:
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

    return true_predictions, true_labels

def compute_metrics(p):
    '''
    Compute accuracy, F1 score, and classification report for the given predictions and labels.

    This function takes a tuple of predictions and labels, converts the predictions to labels, filters out special tokens,
    and then computes the accuracy, F1 score, and classification report. The results are returned in a dictionary.

    Args:
        p (tuple): A tuple containing two elements:
            - predictions (numpy.ndarray): A 2D array of predicted probabilities for each label.
            - labels (list of list): A list of lists, each containing the true labels for a sentence.

    Returns:
        results (dict): A dictionary containing the following metrics:
            - 'accuracy': The accuracy of the predictions.
            - 'f1': The F1 score of the predictions.
            - 'classification_report': The classification report, computed in 'strict' mode with the IOB2 scheme.
    '''
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
          'classification_report': classification_report(true_labels, true_predictions, mode='strict', scheme= IOB2, output_dict=True)    
    }
    return results

def compute_objective(predictions, labels, label_list):
    '''
    Compute the F1 score between the true labels and the predicted labels.

    This function first converts the predictions to the labels using the argmax function.
    It then removes any special tokens that are represented by -100 in the labels.
    Finally, it computes and returns the F1 score between the true labels and the predicted labels.

    Args:
    predictions (np.array): The predicted probabilities for each label. 
                            This is a 2D array where the first dimension is the number of examples and the second dimension is the number of possible labels.
    labels (list of list of int): The true labels for each example. 
                                  This is a list of lists where each inner list contains the labels for one example.
    label_list (list): A list of labels corresponding to the indices in the predictions.

    Returns:
    float: The F1 score between the true labels and the predicted labels.
    '''

    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions, true_labels = generate_true_predictions_and_labels(predictions, labels, label_list)
    
    return f1_score(true_labels, true_predictions)

def generate_csv_comparison(path_data, type_metrics = ['sk', 'strict', 'lenient']):
    '''
    Generate csv files to compare the results reported with classification_report using different libraries (sklearn/seqeval)
    with the results reported in https://aclanthology.org/2022.wiesp-1.4.pdf

    This function reads the classification reports from csv files, calculates the mean, standard deviation, and maximum of the F1 scores,
    and compares these statistics with the results reported in the paper. The results are saved in a new csv file.

    Args:
        path_data (str): Path to the directory where the input csv files are located and where the output csv files will be saved.
        type_metrics (list of str): List of the types of metrics to be computed. The options are 'sk', 'strict', and 'lenient'. 
                                     The function will look for csv files that start with these strings in the path_data directory.

    Returns:
        None. The function saves the results in csv files in the path_data directory.
    '''
    for t in type_metrics: 

        #Read the data
        files = [file for file in os.listdir(path_data) if file.startswith(f"{t}_test_report")]
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        f1_scores = pd.DataFrame()
        for file in files:
            data = pd.read_csv(f"{path_data}/{file}", delimiter=',')
            f1_scores[file] = data['f1-score'].round(4)

        mean = f1_scores.mean(axis=1).round(4)
        sd = f1_scores.std(axis=1).round(4)
        max = f1_scores.max(axis=1).round(4)
    
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
        if t != 'sk':
            f1_scores = f1_scores.reindex(['total-participants', 'intervention-participants','control-participants', 'age', 'eligibility', 'ethinicity', 'condition', 'location', 'intervention', 'control', 'outcome', 'outcome-Measure', 'iv-bin-abs', 'cv-bin-abs', 'iv-bin-percent', 'cv-bin-percent', 'iv-cont-mean', 'cv-cont-mean', 'iv-cont-median', 'cv-cont-median', 'iv-cont-sd', 'cv-cont-sd', 'iv-cont-q1', 'cv-cont-q1', 'iv-cont-q3', 'cv-cont-q3', 'micro avg', 'macro avg', 'weighted avg'])
        else:
            f1_scores = f1_scores.reindex(['total-participants', 'intervention-participants','control-participants', 'age', 'eligibility', 'ethinicity', 'condition', 'location', 'intervention', 'control', 'outcome', 'outcome-Measure', 'iv-bin-abs', 'cv-bin-abs', 'iv-bin-percent', 'cv-bin-percent', 'iv-cont-mean', 'cv-cont-mean', 'iv-cont-median', 'cv-cont-median', 'iv-cont-sd', 'cv-cont-sd', 'iv-cont-q1', 'cv-cont-q1', 'iv-cont-q3', 'cv-cont-q3', 'accuracy', 'macro avg', 'weighted avg'])
        
        #Add manually the values of the paper
        biobert = [0.94, 0.85, 0.88, 0.8, 0.74, 0.88, 0.8, 0.76, 0.84, 0.76, 0.81, 0.84, 0.8, 0.82, 0.87, 0.88, 0.81, 0.86, 0.75, 0.79, 0.83, 0.82, 0, 0, 0, 0, None, None, None]
        longformer = [0.95, 0.85, 0.88, 0.87, 0.88, 0.79, 0.79, 0.87, 0.84, 0.81, 0.85, 0.9, 0.82, 0.82, 0.86, 0.85, 0.84, 0.86, 0.69, 0.73, 0.89, 0.89, 0, 0, 0, 0, None, None, None]
    
        f1_scores['BioBERT_paper'] = biobert
        f1_scores['Mean diff BioBERT'] = (f1_scores['Just mean'] - f1_scores['BioBERT_paper']).round(4)
        f1_scores['Max diff BioBERT'] = (f1_scores['Max'] - f1_scores['BioBERT_paper']).round(4)

        f1_scores['Longformer_paper'] = longformer
        f1_scores['Mean diff Longformer'] = (f1_scores['Just mean'] - f1_scores['Longformer_paper']).round(4)
        f1_scores['Max diff Longformer'] = (f1_scores['Max'] - f1_scores['Longformer_paper']).round(4)

        f1_scores = f1_scores.drop(columns=['Just mean'])

        print("Saving file...")
        f1_scores.to_csv(f"{path_data}/f1_scores_{t}.csv", sep=',')

def generate_learning_curves(path_data):
    '''
    Generate learning curves plots and F1-scores evolution during training on validation set.

    This function reads the training log files, extracts the loss and F1-score values, and generates plots of these metrics over epochs.
    It creates a plot for each training log file, showing the training and validation loss over epochs.
    It also creates a single plot showing the F1-score on the validation set over epochs for all training runs.
    The plots are saved as PNG files in the specified directory.

    Args:
        path_data (str): Path to the directory where the training log files are located and where the output PNG files will be saved.

    Returns:
        None. The function saves the plots as PNG files in the path_data directory.
    '''
    #Read the data
    files = [file for file in os.listdir(path_data) if file.startswith("log_history")]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
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
        plt.close()

    #Get the epochs where the weights are restored and the f1-score
    epoch_save = list(eval_f1.idxmax())
    max_f1     = list(eval_f1.max())    
    print(f"Epochs where the parameters have been restored: {epoch_save}")
    print(f"Maximum F1-scores achieved during training: {max_f1}")
    #Plot the eval_f1
    eval_f1.plot(legend=False)
    plt.scatter(np.array(epoch_save), np.array(max_f1), s=5, c='red')
       
    #plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title("F1-score on validation set over epochs")
    plt.savefig(f"{path_data}/eval_f1.png")
    plt.close()

def annotate_samples(dataset, labels, criteria = 'first_label'):
    '''
    Annotate the sentences in the dataset with the predicted labels.

    This function takes a dataset of sentences and a corresponding list of labels, and annotates each sentence with its predicted labels.
    The labels are assigned to the words in the sentences based on the specified criteria: 'first_label' or 'majority'.
    'first_label' assigns the label of the first sub-token after tokenization to each word, while 'majority' assigns the most frequent label in the tokens into which the word has been divided.

    Args:
        dataset (list of dict): A list of dictionaries, each representing a sentence. Each dictionary contains 'tokens' and 'word_ids'.
        labels (list of list): A list of lists, each containing the predicted labels for a sentence.
        criteria (str): The criteria to use to select the label when the words pieces have different labels. 
                        Options are 'first_label' and 'majority'. Default is 'first_label'.

    Returns:
        annotated_sentences (list of list): A list of lists, each containing the annotated labels for a sentence.
    '''
    
    annotated_sentences = []
    for i in range(len(dataset)):
        # get just the tokens different from None
        sentence = dataset[i]
        word_ids = sentence['word_ids']
        sentence_labels = labels[i]
        annotated_sentence = [[] for _ in range(len(dataset[i]['tokens']))]
        for word_id, label in zip(word_ids, sentence_labels):
            if word_id is not None:
                annotated_sentence[word_id].append(label)
        annotated_sentence_filtered = []
        if criteria == 'first_label':
            annotated_sentence_filtered = [annotated_sentence[i][0] for i in range(len(annotated_sentence)) if len(annotated_sentence[i])>0]
        elif criteria == 'majority':
            annotated_sentence_filtered = [max(set(annotated_sentence[i]), key=annotated_sentence[i].count) for i in range(len(annotated_sentence))]

        annotated_sentences.append(annotated_sentence_filtered)
    return annotated_sentences


def generate_reports(predictions, labels, label_list, save_path, i):
    '''
    Generate classification reports for the given predictions and labels.

    This function generates classification reports in lenient mode, strict mode with the IOB2 scheme, and sklearn mode.
    The reports are saved as CSV files in the specified save path.

    Args:
        predictions (np.array): The predicted probabilities for each label. 
        labels (list of list of int): The true labels for each example.
        label_list (list): A list of labels corresponding to the indices in the predictions.
        save_path (str): The path where the report CSV files will be saved.
        i (int): The index of the current iteration.

    Returns:
        None
    '''
    # Remove ignored index (special tokens)
    true_predictions, true_labels = generate_true_predictions_and_labels(predictions, labels, label_list)

    # IO format
    class_report_lenient = classification_report(true_labels, true_predictions, output_dict=True)
    pd.DataFrame(class_report_lenient).transpose().to_csv(f"{save_path}/lenient_test_report_{i}.csv")

    # IOB2 format
    class_report_strict = classification_report(true_labels, true_predictions, mode = 'strict', scheme=IOB2, output_dict=True)
    pd.DataFrame(class_report_strict).transpose().to_csv(f"{save_path}/strict_test_report_{i}.csv")

    # Remove ignored index (special tokens) and outside tokens
    sk_true_predictions, sk_true_labels = generate_true_predictions_and_labels(predictions, labels, label_list, mode='sklearn')

    sk_true_labels = [label.split('-',1)[1] if len(label.split('-')) > 1 else label for full_true in sk_true_labels for label in full_true]
    sk_true_predictions = [label.split('-',1)[1] if len(label.split('-')) > 1 else label for full_pred in sk_true_predictions for label in full_pred]

    # Lenient mode
    class_report_sk = classification_report_sk(sk_true_labels, sk_true_predictions, output_dict=True)
    pd.DataFrame(class_report_sk).transpose().to_csv(f"{save_path}/sk_test_report_{i}.csv")



BATCH_SIZE = 8

if __name__ == '__main__':
    #Define the argument parser
    parser = argparse.ArgumentParser(description='Fine-tune a model on NER task')
    parser.add_argument('-mode', '--mode', help='Mode to run the script train/test', required=True)
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

    if args['mode'] == 'test':
        MODE = 'test'
    elif args['mode'] == 'train':
        MODE = 'train'

    model_name = model_checkpoint.split("/")[-1]
    data_name = args['data'].split("/")[-1]

    if args['save_path'] != None:
        save_path = f"{args['save_path']}/{model_name}-finetuned-{task}"
    else:
        save_path = f"./{model_name}-finetuned-{task}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    #Load the dataset
    if MODE != "inference":
        print("LOADING DATASET...")
        data_files = {'train':f"train-00000-of-00001.parquet", 
                    'valid': f"valid-00000-of-00001.parquet", 
                    'test': f"test-00000-of-00001.parquet"}
        datasets = load_dataset('parquet', data_dir = args['data'], data_files=data_files)
        print(datasets)

        label_list = datasets['train'].features[f"{task}_tags"].feature.names
        idsxlabel = {i: label for i, label in enumerate(label_list)}
        labelxids = {label: i for i, label in enumerate(label_list)}
        

        #Preprocessing the data
        #Define the tokenizer
        if model_checkpoint == '../models/longformer-base-4096':
            tokenizer = LongformerTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
            flag_tokenizer = 'LongFormer'
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            flag_tokenizer = None

        label_all_tokens = True
        print("TOKENIZING...")
        tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)



    if MODE == 'train':
        #Hyperparameter fine-tuning
        
        print("FINE-TUNING HYPERPARAMETERS...")

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
        def objective(trial: optuna.Trial):
            model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), label2id=labelxids, id2label=idsxlabel)
            print(f"Trial {trial.number}")   
            model.to(device)

            args = TrainingArguments(
            save_path+"/HyperparametersSearch",
            learning_rate=trial.suggest_float("learning_rate", low=2e-5, high=5e-5, log=True),
            weight_decay=trial.suggest_float("weight_decay", 4e-5, 0.01, log=True),
            per_device_train_batch_size= BATCH_SIZE,#trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
            per_device_eval_batch_size= BATCH_SIZE, #trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32]),
            num_train_epochs=10,
            evaluation_strategy = "epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            greater_is_better = False,
            eval_accumulation_steps=1,
            )

            data_collator = DataCollatorForTokenClassification(tokenizer)

            trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['valid'],
            data_collator=data_collator,
            tokenizer=tokenizer,
            )
        
            trainer.train()
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

            #We want to maximize the f1-score in validation set
            predictions, labels, metrics = trainer.predict(tokenized_datasets['valid'])  
            print(f"Validation set metrics: \n {metrics}")
            f1_value = compute_objective(predictions, labels, label_list)
            print(f"F1-Score: {f1_value}")
            return f1_value
            

        # We want to maximize the f1-score in validation set
        study = optuna.create_study(study_name="hyper-parameter-search", direction="maximize")
        study.optimize(func=objective, n_trials=15)
        print(f"Best F1-score: {study.best_value}")
        print(f"Best parameters: {study.best_params}")
        print(f"Best trial: {study.best_trial}")

        file = open(f"{save_path}/best_params.pkl", "wb")
        pickle.dump(study.best_params, file)
        file.close()
        
    
        #TRAINING
        seed = 42
        random.seed(seed)
        
        print("TRAINING...")
        for i in range(5):

            #Define the model
            model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), label2id=labelxids, id2label=idsxlabel)
    
            model.to(device)

            model_name = model_checkpoint.split("/")[-1]
            args = TrainingArguments(
                f"{save_path}/Training_{i}",
                learning_rate= float(study.best_params['learning_rate']), #2e-5
                per_device_train_batch_size= BATCH_SIZE,#int(study.best_params['per_device_train_batch_size']),
                per_device_eval_batch_size= BATCH_SIZE, #int(study.best_params['per_device_eval_batch_size']),
                weight_decay = float(study.best_params['weight_decay']), #0.01
                num_train_epochs=20,
                evaluation_strategy = "epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                metric_for_best_model="f1",
                load_best_model_at_end=True,
                save_total_limit=1,
                seed=random.randint(0,200),
                eval_accumulation_steps=1,
            )

            data_collator = DataCollatorForTokenClassification(tokenizer)

            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['valid'],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            #Train the model
            print(f"Training iteration {i+1}")
            outputs_train = trainer.train()
            if device.type == 'cuda':
                #print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

            print(outputs_train)
            torch.save(model, f"{save_path}/model_{i}.pt") #Save the model
            pd.DataFrame(trainer.state.log_history).to_csv(f"{save_path}/log_history_{i}.csv") #Save the logs of the training

            #Evaluate the model
            print("EVALUATING...")
            predictions, labels, _ = trainer.predict(tokenized_datasets["test"])

            predictions = np.argmax(predictions, axis=2)
            generate_reports(predictions, labels, label_list, save_path, i)

         
        generate_csv_comparison(save_path)
        generate_learning_curves(save_path)


    if MODE == 'test':
        
        input_ids = torch.tensor(tokenized_datasets["test"]["input_ids"]).to(device)
        attention_mask = torch.tensor(tokenized_datasets["test"]["attention_mask"]).to(device)     
        test_data = {'input_ids': input_ids, 'attention_mask': attention_mask}

        labels = tokenized_datasets['test']['labels']

        models_predictions = []
        for i in range(5):
            print(f"EVALUATING... model {i}")

            model = torch.load(f"{save_path}/model_{i}.pt")
            model.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(**test_data)

            predictions = torch.argmax(outputs.logits, dim=2).to("cpu").numpy()
            torch.cuda.empty_cache()
            
            generate_reports(predictions, labels, label_list, save_path, i)

            annotated_samples_first = annotate_samples(tokenized_datasets["test"], predictions, criteria='first_label')
            models_predictions.append(annotated_samples_first)

        generate_csv_comparison(save_path)
        
        #Save predictions for the models in csv files
        dataset_annotated = Dataset.from_dict({'tokens': tokenized_datasets["test"]['tokens'], 
                                                'Predicted_label_0': models_predictions[0],
                                                'Predicted_label_1': models_predictions[1], 
                                                'Predicted_label_2': models_predictions[2],
                                                'Predicted_label_3': models_predictions[3],
                                                'Predicted_label_4': models_predictions[4],  
                                                'file': tokenized_datasets["test"]['id'],
                                                'true_labels': datasets['test']['ner_tags']
                                                })


        # generate the files
        if not  os.path.exists(f"{save_path}/Evaluations"):
            os.makedirs(f"{save_path}/Evaluations")
        most_common_predictions = []
        for j in range(len(dataset_annotated['file'])):
            csv_dict = {'token': [], 'True label': [], 'Most common': [], 'Pred model 1': [], 'Pred model 2': [], 'Pred model 3': [], 'Pred model 4': [], 'Pred model 5': []}
            file_data = dataset_annotated.filter(lambda x: x['file'] == str(j))
            
            
            for sentence, or_labels, preds_1, preds_2, preds_3, preds_4, preds_5 in zip(file_data['tokens'], file_data['true_labels'], file_data['Predicted_label_0'], file_data['Predicted_label_1'], file_data['Predicted_label_2'], file_data['Predicted_label_3'], file_data['Predicted_label_4']):
                for token, or_label, pred_1, pred_2, pred_3, pred_4, pred_5 in zip(sentence, or_labels, preds_1, preds_2, preds_3, preds_4, preds_5):
                    csv_dict['token'].append(token)
                    csv_dict['True label'].append(label_list[or_label])
                    csv_dict['Pred model 1'].append(label_list[pred_1])
                    csv_dict['Pred model 2'].append(label_list[pred_2])
                    csv_dict['Pred model 3'].append(label_list[pred_3])
                    csv_dict['Pred model 4'].append(label_list[pred_4])
                    csv_dict['Pred model 5'].append(label_list[pred_5])
                    occurence_counter = Counter([csv_dict['Pred model 1'][-1], csv_dict['Pred model 2'][-1], csv_dict['Pred model 3'][-1], csv_dict['Pred model 4'][-1], csv_dict['Pred model 5'][-1]])
                    csv_dict['Most common'].append(occurence_counter.most_common(1)[0][0])
                
            most_common_predictions.append(csv_dict['Most common'])

            pd.DataFrame.from_dict(csv_dict,  orient='index').transpose().to_csv(f"{save_path}/Evaluations/File_{j}_IOB.csv")
    

        #Compute the classification reports with most common predictions
        generate_reports(most_common_predictions, labels, label_list, save_path, 'most_common')

                
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    

###TODO: INFERENCE MODE

