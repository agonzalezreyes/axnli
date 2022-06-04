import numpy as np
import json
import torch
import os
import gc
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
from datasets import load_metric


gc.collect()


splits = {'americas_nli': ['validation', 'test'], 'xnli':['train','test', 'validation']}

languages = {'americas_nli': ['aym', 'bzd', 'cni', 'gn', 'hch', 'nah', 'oto', 'quy', 'shp', 'tar', 'all_languages'],
             'xnli':['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh', 'all_languages']}

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mnli")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    results = metric.compute(predictions=predictions, references=labels)
    return results
  
def predict_labels(trainer,tokenized_datasets):
    predictions = trainer.predict(tokenized_datasets)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = load_metric("glue", "mnli")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    results['predictions'] = " ".join([str(i) for i in preds])
    results['gold'] = " ".join([str(i) for i in predictions.label_ids])
    return results



def experiment( exp_dir='',model_name = "xlm-roberta-base", 
                fine_tune_corpus= 'xnli',
                fine_tune_language = 'all_languages',
                eval_corpus='americas_nli',
                eval_language = 'all_languages',
                metric_for_best_model='accuracy',
                learning_rate = 2e-5,
                batch_size = 32,
                num_labels = 3,
                num_train_epochs = 5.0,
                logging_steps = 2500,
                save_steps = 2500,
                save_total_limit = 15,
                patience=15,
                evaluation_strategy= 'steps',
                load_best_model_at_end=True,
                seed = 42, run=1,
                mode = 'train',
                eval_set='validation',
                gradient_accumulation_steps=1,
                gradient_checkpointing=False):
    assert fine_tune_language in languages[fine_tune_corpus]
    torch.cuda.empty_cache()

    if exp_dir=='':
        exp_dir = f'{os.getcwd()}/{fine_tune_language}'
    if not os.path.exists(exp_dir):
        os.system(f'mkdir {exp_dir}')
    if not os.path.exists(f'{exp_dir}/{run}'):
        os.system(f'mkdir {exp_dir}/{run}')

    experiment_run = f'{exp_dir}/{run}'
    best_model = f'{experiment_run}/best_model'

    if mode == 'train':
        
        
        
        #  tokenize your dataset
        dataset = load_dataset(fine_tune_corpus,fine_tune_language)
        tokenize_function = lambda example:  tokenizer(example["premise"], example["hypothesis"], truncation=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir = experiment_run,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            learning_rate = learning_rate,
            per_device_eval_batch_size = batch_size,
            save_total_limit = save_total_limit,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            gradient_checkpointing = gradient_checkpointing,
            num_train_epochs = num_train_epochs,
            logging_steps = logging_steps,
            save_steps = save_steps,
            seed = seed)
    
    
        # Initialize model 
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
       
        # Specify trainer arguments 
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)])
    
        # Begin fine-tuning and save best model 
        trainer.train()
       
        trainer.save_model(best_model)
        torch.cuda.empty_cache()
        
    if mode == 'evaluate':
        dataset = load_dataset(fine_tune_corpus,fine_tune_language)
        

        tokenize_function = lambda example:  tokenizer(example["premise"], example["hypothesis"], truncation=True)
        tokenizer = AutoTokenizer.from_pretrained(best_model)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        #print(dataset)
        
        fine_tune_training_args = TrainingArguments(best_model)
        fine_tune_model = AutoModelForSequenceClassification.from_pretrained(best_model, num_labels=num_labels)

        # Specify trainer arguments 
        trainer = Trainer(
            fine_tune_model,
            fine_tune_training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)])
        
        xnli_acc = predict_labels(trainer = trainer,
               tokenized_datasets=tokenized_datasets[eval_set])
        
        results_xnli = {fine_tune_language:xnli_acc}
        
        
 
        with open (f'{experiment_run}/{fine_tune_corpus}.{eval_set}.results.json','w') as json_file:
            json.dump(results_xnli,json_file)
           
        results_americas_nli = {}
        for lg in languages['americas_nli']:
            americas_nli = load_dataset(eval_corpus,lg, split=eval_set)
            tokenized_americas_nli = americas_nli.map(tokenize_function, batched=True)
            accuracy = predict_labels(trainer = trainer,tokenized_datasets=tokenized_americas_nli)
            results_americas_nli[lg] = accuracy
        
        with open (f'{experiment_run}/{eval_corpus}.{eval_set}.results.json','w') as json_file: 
            json.dump(results_americas_nli,json_file)
 
