import argparse
import os
import logging
import time
import pickle
from shutil import rmtree
from tqdm import tqdm
import random
import copy
import torch
from torch.utils.data import DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AdamW, AutoModelForSeq2SeqLM, T5Tokenizer, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset, ABSADataModule
from eval_utils import compute_scores
from parser import init_args
from modeling import PLText2TextModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# initialization
args = init_args()

wandb.init(project="PROJECT_NAME", entity="USER_NAME")
wandb_logger = WandbLogger(project="PROJECT_NAME")


def get_special_token_list(args):
    if args.is_extra_id == False:
        special_token4aspect = '<ASPECT>'
        special_token4opinion = '<OPINION>'
        special_token4category = '<CATEGORY>'
        special_token4sentiment = '<SENTIMENT>'
        special_token4seperate = '<SEP>'
    else:
        special_token4aspect = '<extra_id_0>'
        special_token4opinion = '<extra_id_1>'
        special_token4category = '<extra_id_3>'
        special_token4sentiment = '<extra_id_2>'
        special_token4seperate = '<extra_id_4>'
    special_token_list = [special_token4aspect, special_token4opinion, special_token4category, special_token4sentiment, special_token4seperate]
    return special_token_list


if __name__ == "__main__":
    ###
    start_time = time.process_time()

    print("\n", "="*30, f"NEW EXP: {args.task} on {args.dataset}", "="*30, "\n")

    seed_everything(args.seed)

    special_token_list = get_special_token_list(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    special_tokens_dict = {'additional_special_tokens': special_token_list}
    tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Here is an example (from the dev set):")

    check_dataset = ABSADataset(tokenizer, 'train', special_token_list, args)
    data_sample = check_dataset[2]  # a random data sample
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=False))
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=False))

    args.save_model_dir = args.output_dir + f'epoch={args.num_train_epochs}' + '/' + args.model_name

    wandb_logger.experiment.config["config/hparams"] = vars(args)
    
    data_module = ABSADataModule(args, tokenizer, special_token_list)
    data_module.setup(stage="fit")
    pl_model = PLText2TextModel(vars(args), tokenizer = tokenizer, special_token_list = special_token_list, train_dataset_size = len(data_module.train_dataset))   
    
    early_stop_callback = EarlyStopping(monitor='val/epoch/f1', mode = 'max', patience = 3)
    checkpoint_callback = ModelCheckpoint(monitor = 'val/epoch/f1',
                                            dirpath = args.save_model_dir,
                                            filename = 'epoch{epoch:02d}-val_f1{val/epoch/f1:.2f}',
                                            save_top_k = 1,
                                            mode = 'max',
                                            auto_insert_metric_name = False,
                                            save_weights_only = False)
    train_params = dict(
        default_root_dir=args.output_dir,
        accelerator = 'gpu',
        devices = 1,
        enable_checkpointing = True,
        callbacks = [checkpoint_callback] if args.task != 'ABSC' else [], # early_stop_callback, 
        check_val_every_n_epoch = 1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        logger = wandb_logger,
        fast_dev_run = args.fast_dev_run, # used to debug code
        track_grad_norm=2,  # track gradient norm
        precision = 16 if args.mixed_precision_training else 32,
        strategy = "ddp" if args.multi_gpu else None,
        auto_select_gpus=True,
        # profiler = "simple", # used to find training loop bottlenecks
        # deterministic=True, # new features in PL 1.6.3 comparing early version
        # limit_train_batches=0.1
    )
    trainer = pl.Trainer(**train_params)

    training_time = ''

    if args.do_train:
        print("\n****** Conduct Training ******")

        trainer.fit(pl_model, datamodule = data_module)
       
        # save the final model
        trainer.save_checkpoint(f"{args.save_model_dir}/last.ckpt")
        
        pl_model.seq2seq_model.save_pretrained(args.save_model_dir)
        pl_model.tokenizer.save_pretrained(args.save_model_dir)

        # record running time
        end_time = time.process_time()
        training_time = str(end_time - start_time)
        print('Training time: ', training_time)

        print("Finish training and saving the model!")

    if args.do_direct_eval:
        # model = PLText2TextModel.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path)
        print(f'Best model path: {checkpoint_callback.best_model_path}')
        print(f'Best val f1 scores: {checkpoint_callback.best_model_score}')

        path_file = open(f"{args.save_model_dir}/best_model_path.txt", 'w', encoding = 'utf-8')
        path_file.write(f"{checkpoint_callback.best_model_path} {checkpoint_callback.best_model_score}\n")
        path_file.close()

        # best valid f1
        pl_model = PLText2TextModel.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path if args.task != 'ABSC' else f"{args.save_model_dir}/last.ckpt")
        prediction_dicts = trainer.test(pl_model, data_module)

        # save performance and hyperparameters to file
        log_file_path = f"./results_log/{args.task}/{args.dataset}_{args.output_type}.txt"
        local_time = time.asctime(time.localtime(time.time()))

        args.device_name = torch.cuda.get_device_name(torch.cuda.current_device())

        exp_base_settings = f"Device: {args.device_name}, Task: {args.task}, Datset={args.dataset}, PTM Version={args.model_name_or_path}, Seed={args.seed}, Train bs={args.train_batch_size}, num_epochs={args.num_train_epochs}, max_input_len={args.max_input_seq_length}, lr={args.learning_rate}, warmup_rate={args.warmup_rate}"
        exp_results = f"Precision={prediction_dicts[0]['test/precision']:.4f}, Recall={prediction_dicts[0]['test/recall']:.4f}, F1 = {prediction_dicts[0]['test/f1']:.4f}"

        log_str = f'============================================================\n'
        if training_time != '':
            log_str += f"Training time: {training_time}\n"
        log_str += f"{local_time}\n{exp_base_settings}\n{exp_results}\n\n"

        if not os.path.exists(f'./results_log/{args.task}'):
            os.mkdir(f'./results_log/{args.task}')

        with open(log_file_path, "a+") as f:
            f.write(log_str)
        
        # rmtree(args.save_model_dir)


    if args.do_inference:
        if args.is_load_default_model4inference:
            model = PLText2TextModel(vars(args), tokenizer = tokenizer, special_token_list = special_token_list, train_dataset_size = len(data_module.train_dataset))
            predictions = trainer.test(model, data_module)
        else:
            print("\n********* Conduct inference on trained checkpoint *********")
            print(f"Load trained model from {args.save_model_dir}")

            print("\n*** last epoch checkpoint performance ***")
            if args.load_model == 'torch':
                args.model_name_or_path = args.save_model_dir
                model = PLText2TextModel(vars(args), tokenizer, special_token_list)
            else:
                model = PLText2TextModel.load_from_checkpoint(checkpoint_path = f"{args.save_model_dir}/last.ckpt")
            trainer.validate(model, data_module)
            trainer.test(model, data_module)

            print("\n*** best val f1 checkpoint performance ***")
            with open(f"{args.save_model_dir}/best_model_path.txt", 'r', encoding='UTF-8') as fp:
                for line in fp:
                    best_model_path, val_f1_scores = line.strip().split(' ')
                    print(f"Load trained model (val f1 best) from {best_model_path}")
                    print(f"Best f1 score on val set: {val_f1_scores}")
            model = PLText2TextModel.load_from_checkpoint(checkpoint_path = best_model_path)
            predictions = trainer.test(model, data_module)
            print('*'*100)
            print(predictions)

        # save performance and hyperparameters to file
        log_file_path = f"./results_log/{args.task}/{args.dataset}_{args.output_type}.txt"
        local_time = time.asctime(time.localtime(time.time()))

        args.device_name = torch.cuda.get_device_name(torch.cuda.current_device())

        exp_base_settings = f"Device: {args.device_name}, Task: {args.task}, Datset={args.dataset}, PTM Version={args.model_name_or_path}, Seed={args.seed}, Train bs={args.train_batch_size}, num_epochs={args.num_train_epochs}, max_input_len={args.max_input_seq_length}, lr={args.learning_rate}, warmup_rate={args.warmup_rate}"
        exp_results = f"Precision={predictions[0]['test/precision']:.4f}, Recall={predictions[0]['test/recall']:.4f}, F1 = {predictions[0]['test/f1']:.4f}"

        log_str = f'============================================================\n'
        if training_time != '':
            log_str += f"Training time: {training_time}\n"
        log_str += f"{local_time}\n{exp_base_settings}\n{exp_results}\n\n"

        if not os.path.exists(f'./results_log/{args.task}'):
            os.mkdir(f'./results_log/{args.task}')

        with open(log_file_path, "a+") as f:
            f.write(log_str)