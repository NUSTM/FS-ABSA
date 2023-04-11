import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl



sentiment = ['negative', 'neutral', 'positive']
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
aspect_cate_list_rest16 = ['restaurant miscellaneous', 'location general', 'food prices', 'drinks quality', 'restaurant general', 'drinks prices', 'ambience general', 'drinks style options', 'service general', 'food general', 'restaurant prices', 'food style options', 'food quality']
aspect_cate_list_laptop = ['ports design features', 'hard disc miscellaneous', 'laptop connectivity', 'hard disc price', 'shipping price', 'os quality', 'mouse general', 'display general', 'memory general', 'os miscellaneous', 'motherboard general', 'power supply quality', 'graphics operation performance', 'display price', 'hardware quality', 'optical drives design features', 'keyboard price', 'software usability', 'display quality', 'multimedia devices design features', 'hard disc quality', 'support general', 'display operation performance', 'motherboard quality', 'laptop design features', 'fans&cooling design features', 'support operation performance', 'display usability', 'optical drives general', 'company general', 'software design features', 'out of scope general', 'memory operation performance', 'laptop price', 'support design features', 'software general', 'keyboard usability', 'support quality', 'keyboard quality', 'os general', 'power supply operation performance', 'power supply general', 'laptop usability', 'ports portability', 'display design features', 'hard disc operation performance', 'laptop portability', 'shipping quality', 'os operation performance', 'mouse usability', 'os price', 'cpu operation performance', 'hard disc general', 'optical drives usability', 'company quality', 'os design features', 'os usability', 'hardware operation performance', 'hardware design features', 'ports connectivity', 'multimedia devices quality', 'laptop quality', 'company operation performance', 'mouse design features', 'cpu general', 'hard disc design features', 'keyboard design features', 'company price', 'software quality', 'multimedia devices general', 'out of scope design features', 'ports quality', 'multimedia devices price', 'shipping operation performance', 'cpu quality', 'multimedia devices usability', 'memory quality', 'memory usability', 'cpu price', 'hardware usability', 'multimedia devices operation performance', 'graphics usability', 'company design features', 'power supply connectivity', 'warranty general', 'laptop general', 'hard disc usability', 'fans&cooling general', 'hardware general', 'multimedia devices connectivity', 'keyboard general', 'keyboard portability', 'battery design features', 'keyboard miscellaneous', 'fans&cooling quality', 'motherboard operation performance', 'keyboard operation performance', 'power supply design features', 'support price', 'software price', 'laptop operation performance', 'graphics general', 'out of scope operation performance', 'warranty quality', 'cpu design features', 'laptop miscellaneous', 'out of scope usability', 'software operation performance', 'software portability', 'battery general', 'battery quality', 'graphics design features', 'memory design features', 'hardware price', 'ports operation performance', 'shipping general', 'ports usability', 'battery operation performance', 'ports general', 'fans&cooling operation performance', 'optical drives operation performance']


def read_line_examples_from_file(data_path, silence=True):
    """
    Read unified format data from file, each line is "sentence####[["aspect", "category", "semtiment", "opinion"], [...]]"
    Return List[List[word]], List[Quad]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_span_inputs(sents, labels, special_token_list, task):
    inputs = []
    special_token_aspect, special_token_opinion, special_token_category, special_token_sentiment, special_token_seperate = special_token_list
    
    for words_list, label in zip(sents, labels):
        words = ' '.join(words_list)

        if task == 'ASPE':
            template_words = f"{special_token_aspect} is {special_token_sentiment}."
        
        content = words + ' ' + template_words
        content_list = content.split(' ')
        inputs.append(content_list)
    return [s.copy() for s in inputs]


def get_aspe_targets(labels, special_token_list, output_type):
    targets = []
    special_token_aspect, special_token_opinion, special_token_category, special_token_sentiment, special_token_seperate = special_token_list

    for label in labels:
        all_pair_target = []
        for pair in label:
            aspect, sentiment = pair    
            if output_type == 'span': 
                one_pair_target = f"{special_token_aspect} {aspect} {special_token_sentiment} {sentword2opinion[sentiment]}"
            elif output_type == 'paraphrase': 
                one_pair_target = f"{aspect} is {sentiment}"
            elif output_type == 'extraction':
                one_pair_target = f"({aspect}, {sentiment})"
            
            all_pair_target.append(one_pair_target)

        if output_type == 'span':
            target = f" ".join(all_pair_target)
        elif output_type == 'paraphrase':
            target = ' [SSEP] '.join(all_pair_target)
        elif output_type == 'extraction':
            target = '; '.join(all_pair_target)
        targets.append(target)
    return targets


def get_transformed_io(data_path, data_dir, task='ACOSQE', special_token_list = None, output_type='span'):
    """
    The main function to transform input & target according to the task
    Return List[List[sentence template]], List[String(ground truth)]
    """
    sents, labels = read_line_examples_from_file(data_path)


    if task == 'ASPE':
        if output_type == 'span':
            inputs = get_span_inputs(sents, labels, special_token_list, task)
            targets = get_aspe_targets(labels, special_token_list, output_type)
        elif output_type == 'paraphrase':
            inputs = [s.copy() for s in sents]
            targets = get_aspe_targets(labels, special_token_list, output_type)
        elif output_type == 'extraction':
            inputs = [s.copy() for s in sents]
            targets = get_aspe_targets(labels, special_token_list, output_type)
    else:
        raise NotImplementedError

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_type, special_token_list = None, args = None):
        self.max_input_len = args.max_input_seq_length
        self.max_output_len = args.max_output_seq_length
        self.tokenizer = tokenizer
        self.dataset_name = args.dataset
        self.data_type = data_type
        self.task = args.task
        self.special_token_list = special_token_list
        self.output_type = args.output_type
        self.few_shot_data = args.few_shot_data
        self.use_x_shot = args.use_x_shot
        self.use_french_data = args.use_french_data
        self.use_dutch_data = args.use_dutch_data

        if self.use_french_data:
            if self.few_shot_data != 0 and self.use_x_shot != 0:
                self.data_path = f'./data4fewshot/{self.use_x_shot}shot/{self.task}/{self.dataset_name}/{self.data_type}_{self.few_shot_data}.txt'
            else:
                self.data_path = f'./data4ml/french/{self.dataset_name}/{self.data_type}.txt'
        elif self.use_dutch_data:
            if self.few_shot_data != 0 and self.use_x_shot != 0:
                self.data_path = f'./data4fewshot/{self.use_x_shot}shot/{self.task}/{self.dataset_name}/{self.data_type}_{self.few_shot_data}.txt'
            else:
                self.data_path = f'./data4ml/dutch/{self.dataset_name}/{self.data_type}.txt'
        elif self.few_shot_data != 0 and self.use_x_shot != 0:
            self.data_path = f'./data4fewshot/{self.use_x_shot}shot/{self.task}/{self.dataset_name}/{self.data_type}_{self.few_shot_data}.txt' 
        else:
            self.data_path = f'./data/{self.task}/{self.dataset_name}/{self.data_type}.txt'

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):
        # inputs: List[List[sentence template]]
        # targets: List[String(ground truth)]
        inputs, targets = get_transformed_io(self.data_path, self.dataset_name, self.task, self.special_token_list, self.output_type)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_input_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_output_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


class ABSADataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer, special_token_list):
        super().__init__()
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.special_token_list = special_token_list
        self.tokenizer = tokenizer
        self.args = args

    def prepare_data(self):
        # do not use assign operation in prepare_data, such as self.x = y
        pass
    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = ABSADataset(self.tokenizer, 'train', self.special_token_list, self.args)
        if stage == 'fit' or stage == 'validate':
            self.val_dataset = ABSADataset(self.tokenizer, 'dev', self.special_token_list, self.args)
        if stage == 'test' or stage == 'predict':
            self.test_dataset = ABSADataset(self.tokenizer, 'test', self.special_token_list, self.args)
    
    def train_dataloader(self,):
        return DataLoader(self.train_dataset, batch_size= self.train_batch_size, shuffle=True, num_workers = 40, drop_last = True)

    def val_dataloader(self,):
        return DataLoader(self.val_dataset, batch_size = self.eval_batch_size, num_workers = 40)

    def test_dataloader(self,):
        return DataLoader(self.test_dataset, batch_size = self.eval_batch_size, num_workers = 40)

    def predict_dataloader(self,):
        return DataLoader(self.test_dataset, batch_size = self.eval_batch_size, num_workers = 40)
