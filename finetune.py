import click
from nli import experiment

'''
Usage: `python finetune.py --language='es' --run=1 --seed=42`

To use gradient checkpointing, pass the flag --grad_checkpointing and a value for --grad_accumulation_steps,
such as:
    `python finetune.py --language='es' --run=1 --seed=42 --grad_checkpointing --grad_accumulation_steps=1`
'''

@click.command()
@click.option('-l', '--language', type=str, required=True, help="Language to finetune on, e.g. es, en, etc.")
@click.option('-r', '--run', type=int, required=True, help="Run number.")
@click.option('-s', '--seed', type=int, required=True, help="Seed for experiment.")
@click.option('-b', '--batch_size', type=int, default=16, help="Batch size")
@click.option('--save_total_limit', type=int, default=5, help="Total limit of saved checkpoints.")
@click.option('--save_steps', type=int, default=2500, help="Number of steps to save checkpoints.")
@click.option('--epochs', type=int, default=5, help="Number of epochs")
@click.option('--patience', type=int, default=15, help="Patience value")
@click.option('-lr', '--learning_rate', type=float, default=2e-5, help="Learning rate.")
@click.option('--grad_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
@click.option('--grad_checkpointing', is_flag = True, help="Gradient checkpointing")

def main(language, run, seed, batch_size, save_total_limit, \
    save_steps, epochs, patience, learning_rate, grad_accumulation_steps, grad_checkpointing):
    
    # Fine-tune XLM-R using english from XNLI
    experiment(model_name = "xlm-roberta-base",\
        fine_tune_corpus= 'xnli',
        fine_tune_language = language, 
        metric_for_best_model='accuracy',
        learning_rate = learning_rate,
        batch_size = batch_size,
        num_labels = 3,
        num_train_epochs = epochs,
        logging_steps = save_steps,
        save_steps = save_steps,
        save_total_limit = save_total_limit,
        patience=patience,
        evaluation_strategy= 'steps',
        load_best_model_at_end=True,
        seed = seed, # change 42,32,22,12,2
        run=run, # change 1,2,3,4,5
        mode = 'train',
        gradient_accumulation_steps = grad_accumulation_steps,
        gradient_checkpointing = grad_checkpointing)

    eval_set = 'validation'

    # Evaluate XLM-R fine-tuned on english-XNLI
    # output: XNLI accuracy and labels for eval_set         (en/{run}/xnli.{eval_set}.results)
    #         AmericasNLI per language accuracy and labels 
    #         for eval_set                                 (en/{run}/americas_nli.{eval_set}.results)

    experiment(model_name = "xlm-roberta-base", fine_tune_corpus= 'xnli',\
        fine_tune_language = language,eval_corpus='americas_nli',
        eval_language = 'all_languages', metric_for_best_model='accuracy',
        batch_size = batch_size,
        num_labels = 3,
        seed = seed,  # change 42,32,22,12,2
        run=run, # change 1,2,3,4,5
        mode = 'evaluate',
        eval_set=eval_set)

if __name__ == '__main__':
    main()