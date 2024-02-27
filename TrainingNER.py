import os
os.environ['HTTP_PROXY'] = 'http://webproxy.lab-ia.fr:8080'
os.environ['HTTPS_PROXY'] = 'http://webproxy.lab-ia.fr:8080'

#Import required packages
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, PreTrainedTokenizerFast
from datasets import load_dataset, load_metric
import evaluate
import argparse
import torch
from datasets import ClassLabel, Sequence
import random
import numpy as np
import pandas as pd
from IPython.display import display, HTML

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
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, max_length=512, is_split_into_words=True)

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

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

BATCH_SIZE =16

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

    if args['save_path'] != None:
        save_path = args['save_path']
    else:
        model_name = model_checkpoint.split("/")[-1]
        data_name = args['data'].split("/")[-1]
        save_path = f"{model_name}-finetuned-{task}/{data_name}"

    #Load the dataset
    print("LOADING DATASET...")
    datasets = load_dataset(args['data'])
    print(datasets)

    label_list = datasets['train'].features[f"{task}_tags"].feature.names
    idsxlabel = {i: label for i, label in enumerate(label_list)}
    labelxids = {label: i for i, label in enumerate(label_list)}

    #Print some examples of the dataset
    show_random_elements(datasets["train"])

    #Preprocessing the data
    #Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    label_all_tokens = True
    print("TOKENIZING...")
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    #Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    input("Press Enter to continue...")

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    print(model)

    model.to(device)

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        save_path,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    metric = evaluate.load("seqeval")

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
    trainer.train()
    torch.save(model, f'{save_path}/{model_name}.pt')

    #Evaluate the model
    print("EVALUATING...")
    trainer.evaluate(tokenized_datasets["test"])

