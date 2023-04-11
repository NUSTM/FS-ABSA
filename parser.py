import os
import argparse


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='ASPE', type=str, choices=['ASPE'],
                        help="The name of the task, selected from: [ASPE]")
    parser.add_argument("--dataset", default='rest14', type=str, help="The name of the dataset")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--tokenizer_name_or_path", default='t5-base', type=str,
                        help="Path to tokenizer")
    parser.add_argument("--model_name", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--save_model_dir", default='T5')
    parser.add_argument("--device_name", default='NVIDIA GeForce RTX 3090', type=str)
    parser.add_argument("--neptune_mode", default='offline', type=str)

    # other parameters
    parser.add_argument("--max_input_seq_length", default=128, type=int)
    parser.add_argument("--max_output_seq_length", default=128, type=int)
    # changed
    parser.add_argument("--n_gpu", default=[0], type=list)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--mixed_precision_training", action='store_true')
    parser.add_argument("--multi_gpu", action='store_true')

    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,  choices=[7, 8, 9, 26, 27, 28, 29, 42, 51, 80, 120, 33, 153, 757, 969],
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_rate", default=0.1, type=float)   # 0.0/0.1

    # training setting
    parser.add_argument("--training_setting", default='fine-tuning', type=str, choices = ['pre-training', 'fine-tuning'])
    

    # model selection
    parser.add_argument('--model_selection', type=str, default='final_epoch',  choices=["final_epoch", "val_best"],
                        help="Eval which model, i.e. final epoch model or best model on val dataset")
    parser.add_argument('--load_model', type = str, default = 'pl', choices = ['pl', 'torch'], help = "how to load a fine-tuned model to conduct inference, i.e., based on PL or raw Pytorch?")
    # output type
    parser.add_argument('--output_type', type = str, default = 'span', choices = ['paraphrase','extraction', 'span'])

    # output
    parser.add_argument("--save_inference_output",action='store_true')

    # debug code
    parser.add_argument("--fast_dev_run", action = "store_true")

    # type of special token
    parser.add_argument("--is_extra_id", action='store_true')

    # load model for inference
    parser.add_argument("--is_load_default_model4inference", action='store_true')

    # load x-shot data
    parser.add_argument("--use_x_shot", type=int, default=0,  choices=[32, 128, 512])

    # load few shot data
    parser.add_argument("--few_shot_data", type=int, default=0,  choices=[7, 8, 9, 26, 27, 28, 51, 80, 120, 33, 153, 757, 969])

    # do fuzzy matching
    parser.add_argument("--do_fuzzy_matching", action='store_true')

    # use multi-language data
    parser.add_argument("--use_french_data", action='store_true')

    # use multi-language data
    parser.add_argument("--use_dutch_data", action='store_true')

    args = parser.parse_args()


    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    if not os.path.exists('./outputs/fine-tuned/'):
        os.mkdir('./outputs/fine-tuned/')
    
    if not os.path.exists('./outputs/pre-trained'):
        os.mkdir('./outputs/pre-trained')

    
    args.output_dir = create_dirs(args.task, args.dataset, args.output_type, args.seed)

    return args


def create_dirs(task_name, dataset, output_type, seed):
    hier_dict_list = [task_name, dataset, output_type, seed] # order is matter
    meta_path = f'outputs/fine-tuned/'
    for item in hier_dict_list:
        meta_path += f'{item}/'
        if not os.path.exists(meta_path):
            os.mkdir(meta_path)
    return meta_path
