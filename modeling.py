from transformers import AdamW, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl

from data_utils import *
from eval_utils import *


class PLText2TextModel(pl.LightningModule):
    """
    Fine tune or continually pre-train a pre-trained T5 model
    """
    def __init__(self, hparams, tokenizer, special_token_list, train_dataset_size):
        # super(T5FineTuner, self).__init__()
        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(hparams)
        self.seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(hparams['model_name_or_path'])
        # self.f1_metrics = nn.ModuleDict({'train_f1': TupleF1(), 'val_f1': TupleF1(), 'test_f1': TupleF1()})
        self.tokenizer = tokenizer
        self.seq2seq_model.resize_token_embeddings(len(tokenizer))
        self.special_token_list = special_token_list
        self.train_dataset_size = train_dataset_size

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.seq2seq_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"].detach().clone()
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
        )
        return outputs[0]
    
    def _inference_step(self, batch):
        outs = self.seq2seq_model.generate(input_ids = batch["source_ids"], attention_mask = batch["source_mask"], max_length = self.hparams.max_output_seq_length)
        preds = [self.tokenizer.decode(ids, skip_special_tokens = False) for ids in outs]

        targets = [self.tokenizer.decode(ids, skip_special_tokens = False) for ids in batch['target_ids']]
        return preds, targets

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train/batch/loss", loss)
        return {"loss": loss}
        # self.logger.experiment["train/batch/loss"].log(loss)

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train/epoch/avg_loss", avg_train_loss)
    
    def validation_step(self, batch, batch_idx):
        val_loss = self._step(batch)
        self.log("val/batch/loss", val_loss)
        preds, targets = self._inference_step(batch)
        return {"loss":val_loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds, targets = [], []
        for x in outputs:
            preds.extend(x['preds'])
            targets.extend(x['targets'])
        val_scores, _, _ = compute_scores(preds, targets, task = self.hparams.task, output_type = self.hparams.output_type, special_token_list = self.special_token_list, use_x_shot = self.hparams.use_x_shot, few_shot_data = self.hparams.few_shot_data, use_french_data = self.hparams.use_french_data, use_dutch_data = self.hparams.use_dutch_data, dataset = self.hparams.dataset, is_test_mode = 0, do_fuzzy_matching = self.hparams.do_fuzzy_matching)
        log_value_dict = {"val/epoch/avg_loss": avg_loss, "val/epoch/f1": val_scores['f1']}
        self.log_dict(log_value_dict)

    def test_step(self, batch, batch_idx):
        preds, targets = self._inference_step(batch)
        return {"preds": preds, "targets": targets}
    
    def test_epoch_end(self, outputs):
        preds, targets = [], []
        for x in outputs:
            preds.extend(x['preds'])
            targets.extend(x['targets'])
        test_scores, all_pred_tuples, all_target_tuples = compute_scores(preds, targets, task = self.hparams.task, output_type = self.hparams.output_type, special_token_list = self.special_token_list, use_x_shot = self.hparams.use_x_shot, few_shot_data = self.hparams.few_shot_data, use_french_data = self.hparams.use_french_data, use_dutch_data = self.hparams.use_dutch_data, dataset = self.hparams.dataset, is_test_mode = 1, do_fuzzy_matching = self.hparams.do_fuzzy_matching)

        macro_f1 = -1
        if "macro-f1" in test_scores.keys():
            macro_f1 = test_scores["macro-f1"]
        log_value_dict = {"test/precision": test_scores['precision'], "test/recall": test_scores['recall'], "test/f1": test_scores['f1'], "test/macro-f1": macro_f1}
        self.log_dict(log_value_dict)
        return test_scores, all_pred_tuples, all_target_tuples
    
    def predict_step(self, batch, batch_idx):
        preds, targets = self._inference_step(batch)
        return {"preds": preds, "targets": targets}


    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.seq2seq_model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.optimizer = optimizer
        
        t_total = (
            (self.train_dataset_size // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(self.hparams.warmup_rate * t_total), num_training_steps=t_total
        )

        return optimizer

    def optimizer_step(self,
                    epoch=None, 
                    batch_idx=None, 
                    optimizer=None, 
                    optimizer_idx=None, 
                    optimizer_closure=None, 
                    on_tpu=None, 
                    using_native_amp=None, 
                    using_lbfgs=None
                     ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()