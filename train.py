import argparse
import os
import pickle
import json
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from eval_utils import get_final_results
from data_utils import MyDataset, get_examples, get_constrained_decoding_token_ids, get_aug_data, get_task_config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
# device_name = torch.cuda.get_device_name(0)
# print(device_name)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='aste', type=str)
    parser.add_argument("--dataset", default='rest16', type=str)
    parser.add_argument("--model_name_or_path", default='t5-base', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--max_ans_length", default=32, type=int)
    parser.add_argument("--n_gpu", default="0", type=str)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1,  type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--phase_1_epochs", default=5, type=int)
    parser.add_argument("--phase_2_epochs", default=20, type=int)
    parser.add_argument("--num_train_epochs", default=None, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--num_dataloader_workers", default=4, type=int)
    parser.add_argument("--output_dir", default='outputs', type=str)
    parser.add_argument("--cls_weight", default=1, type=float)
    parser.add_argument("--train_file", default='train', type=str)
    parser.add_argument("--dev_file", default='dev', type=str)
    parser.add_argument("--test_file", default='test', type=str)
    parser.add_argument("--do_phase_1", default=True, type=bool)
    parser.add_argument("--do_phase_2", default=True, type=bool)
    parser.add_argument("--do_eval_all_ckpts", default=True, type=bool)
    parser.add_argument("--num_beams", default=6, type=int)
    parser.add_argument("--domain", default="absa", type=str)
    parser.add_argument("--task_config_file", default="task_config.json", type=str)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, f"{args.task}-{args.dataset}")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    return args


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'])
        loss_weight = batch['loss_weight'].view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        logits = outputs[1]
        loss = loss_fct(logits.view(-1, logits.size(-1)), lm_labels.view(-1))
        loss = loss * loss_weight
        loss = loss[loss_weight > 0].mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = MyDataset(tokenizer=self.tokenizer, data_dir=self.hparams.dataset, data_type=self.hparams.train_file, task=self.hparams.task, split_tuples=True, cls_weight=self.hparams.cls_weight, max_len=self.hparams.max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=self.hparams.num_dataloader_workers)
        t_total = ((len(train_dataset) // (self.hparams.train_batch_size * 1)) // self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs))
        scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        self.lr_scheduler = scheduler
        return train_dataloader

    def val_dataloader(self):
        val_dataset = MyDataset(tokenizer=self.tokenizer, data_dir=self.hparams.dataset, data_type=self.hparams.dev_file, task=self.hparams.task, split_tuples=True, cls_weight=self.hparams.cls_weight, max_len=self.hparams.max_seq_length)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_dataloader_workers)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        metrics = None
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        metrics = None
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def train(args):
    model = T5FineTuner(args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=-1, save_weights_only=True)
    train_params = dict(default_root_dir=args.output_dir,
                        accumulate_grad_batches=args.gradient_accumulation_steps,
                        gpus=args.n_gpu,
                        gradient_clip_val=1.0,
                        max_epochs=args.num_train_epochs,
                        checkpoint_callback=checkpoint_callback,
                        callbacks=[LoggingCallback()])
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    model.model.save_pretrained(args.output_dir)
    return model


def evaluate(data_loader, model, args, tokenizer, task_config):
    model.model.to(device)
    model.model.eval()
    constrained_decoding_token_ids = get_constrained_decoding_token_ids(tokenizer, task_config)
    raw_outputs = []
    for batch in tqdm(data_loader):
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            allowed_tokens = batch['source_ids'][batch_id].tolist()
            allowed_tokens += constrained_decoding_token_ids
            allowed_tokens = list(set(allowed_tokens))
            return allowed_tokens
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                    attention_mask=batch['source_mask'].to(device),
                                    max_length=args.max_ans_length,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    num_beams=args.num_beams,
                                    num_return_sequences=args.num_beams,
                                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        batch_size = batch['source_ids'].size(0)
        beam_size = int(len(outs['sequences'])/batch_size)
        for i in range(batch_size):
            j1 = i * beam_size
            j2 = (i+1) * beam_size
            probs = torch.exp(outs['sequences_scores'][j1:j2]).tolist()
            seqs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs['sequences'][j1:j2]]
            raw_output = [(prob, seq) for prob, seq in zip(probs, seqs)]
            raw_outputs.append(raw_output)
    return raw_outputs


def main():
    args = init_args()
    seed_everything(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    task_config = get_task_config(args.task_config_file, args.task, args.dataset)

    if args.do_phase_1:
        print('========== phase 1 ==========')
        # train
        print('========== training ==========')
        args.num_train_epochs = args.phase_1_epochs
        args.output_dir = os.path.join(args.output_dir, 'phase_1')
        os.mkdir(args.output_dir)
        model = train(args)
        # data aug
        data_types = [args.train_file, args.dev_file]
        for data_type in data_types:
            print(f'========== data augmentation for: {data_type} ==========')
            dataset = MyDataset(tokenizer, data_dir=args.dataset, data_type=data_type, task=args.task, split_tuples=False, cls_weight=args.cls_weight, max_len=args.max_seq_length)
            dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=args.num_dataloader_workers)
            raw_outputs = evaluate(dataloader, model, args, tokenizer, task_config)
            sents, targets = get_examples(dataset.data_path, split_tuples=False)
            aug_data = get_aug_data(sents, targets, raw_outputs, task_config)
            with open(f'data/{args.task}/{args.dataset}/{data_type}_aug.txt', 'w') as f:
                f.write('\n'.join(aug_data))
            with open(os.path.join(args.output_dir, f'{data_type}_aug.txt'), 'w') as f:
                f.write('\n'.join(aug_data))

    if args.do_phase_2:
        print('========== phase 2 ==========')
        # train
        print('========== training ==========')
        args.num_train_epochs = args.phase_2_epochs
        args.output_dir = os.path.join(args.output_dir.replace('phase_1', ''), 'phase_2')
        os.mkdir(args.output_dir)
        args.train_file += '_aug'
        args.dev_file += '_aug'
        model = train(args)
        # eval
        print(f'========== evaluating {args.test_file} ==========')
        dataset = MyDataset(tokenizer, data_dir=args.dataset, data_type=args.test_file, task=args.task, split_tuples=False, cls_weight=args.cls_weight, max_len=args.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=args.num_dataloader_workers)
        raw_outputs = evaluate(dataloader, model, args, tokenizer, task_config)
        sents, targets = get_examples(dataset.data_path, split_tuples=False)
        all_results = [(x, y, z) for x, y, z in zip(sents, targets, raw_outputs)]
        pickle.dump(all_results, open(os.path.join(args.output_dir, "test_results.pickle"), 'wb'))
        scores, bad_cases = get_final_results(sents, targets, raw_outputs, task_config)
        with open(os.path.join(args.output_dir, 'scores.json'), "w") as writer:
            writer.write(json.dumps(scores, indent=4) + "\n")
    # eval all
    if args.do_eval_all_ckpts:
        print(f'========== evaluating all ckpts ==========')
        for data_type in ["dev", "test"]:
            print(f'========== {data_type} ==========')
            dataset = MyDataset(tokenizer, data_dir=args.dataset, data_type=data_type, task=args.task, split_tuples=False, cls_weight=args.cls_weight, max_len=args.max_seq_length)
            dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=args.num_dataloader_workers)
            sents, targets = get_examples(dataset.data_path, split_tuples=False)
            all_checkpoints, all_epochs = [], []
            for f in os.listdir(args.output_dir):
                file_name = os.path.join(args.output_dir, f)
                if 'cktepoch' in file_name:
                    all_checkpoints.append(file_name)
            all_scores = []
            for checkpoint in all_checkpoints:
                epoch = checkpoint.split('=')[-1][:-5] if len(checkpoint) > 1 else ""
                print(f"epoch={epoch}")
                model_ckpt = torch.load(checkpoint)
                model = T5FineTuner(model_ckpt['hyper_parameters'])
                model.load_state_dict(model_ckpt['state_dict'])
                raw_outputs = evaluate(dataloader, model, args, tokenizer, task_config)
                scores, _ = get_final_results(sents, targets, raw_outputs, task_config)
                all_scores.append((epoch, scores))
                print(scores)
            with open(os.path.join(args.output_dir, f'all_scores_{data_type}.txt'), "w") as writer:
                best_epoch, best_f1 = -1, -1.0
                for epoch, scores in all_scores:
                    if scores['f1'] > best_f1:
                        best_epoch, best_f1 = epoch, scores['f1']
                    writer.write(f"epoch={epoch}\n")
                    writer.write(json.dumps(scores, indent=4) + "\n")
                writer.write(f"best epoch={best_epoch}, f1={best_f1}\n")


if __name__ == "__main__":
    main()
