import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange
from flask import Flask, request, jsonify

app = Flask(__name__)

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def create_chatbot(id):
    data = pd.read_csv(f'{id}.csv')

    CHARACTER_NAME = data['name'].mode().values[0]

    contexted = []

    n = 7

    for i in data[data.name == CHARACTER_NAME].index:
        if i < n:
            continue
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data.line[j])
        contexted.append(row)

    columns = ['response', 'context']
    columns = columns + ['context/' + str(i) for i in range(n - 1)]

    df = pd.DataFrame.from_records(contexted, columns=columns)

    trn_df, val_df = train_test_split(df, test_size=0.1)
    trn_df.head()

    def construct_conv(row, tokenizer, eos=True):
        flatten = lambda l: [item for sublist in l for item in sublist]
        conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
        conv = flatten(conv)
        return conv

    class ConversationDataset(Dataset):
        def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

            block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

            directory = args.cache_dir
            cached_features_file = os.path.join(
                directory, args.model_type + "_cached_lm_" + str(block_size)
            )

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
            else:
                logger.info("Creating features from dataset file at %s", directory)

                self.examples = []
                for _, row in df.iterrows():
                    conv = construct_conv(row, tokenizer)
                    self.examples.append(conv)

                logger.info("Saving features into cached file %s", cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, item):
            return torch.tensor(self.examples[item], dtype=torch.long)

    def load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False):
        return ConversationDataset(tokenizer, args, df_val if evaluate else df_trn)

    def set_seed(args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
        if not args.save_total_limit:
            return
        if args.save_total_limit <= 0:
            return

        checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
        if len(checkpoints_sorted) <= args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    from transformers import AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

    logger = logging.getLogger(__name__)

    MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
    MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

    class Args():
        def __init__(self, id):
            self.output_dir = id
            self.model_type = 'gpt2'
            self.model_name_or_path = 'microsoft/DialoGPT-small'
            self.config_name = 'microsoft/DialoGPT-small'
            self.tokenizer_name = 'microsoft/DialoGPT-small'
            self.cache_dir = 'cached'
            self.block_size = 512
            self.do_train = True
            self.do_eval = True
            self.evaluate_during_training = False
            self.per_gpu_train_batch_size = 4
            self.per_gpu_eval_batch_size = 4
            self.gradient_accumulation_steps = 1
            self.learning_rate = 5e-5
            self.weight_decay = 0.0
            self.adam_epsilon = 1e-8
            self.max_grad_norm = 1.0
            self.num_train_epochs = 4
            self.max_steps = -1
            self.warmup_steps = 0
            self.logging_steps = 1000
            self.save_steps = 3500
            self.save_total_limit = None
            self.eval_all_checkpoints = False
            self.no_cuda = False
            self.overwrite_output_dir = True
            self.overwrite_cache = True
            self.should_continue = False
            self.seed = 42
            self.local_rank = -1
            self.fp16 = False
            self.fp16_opt_level = 'O1'

    args = Args(id)

    def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
        """ Train the model """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate, drop_last=True
        )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        model = model.module if hasattr(model, "module") else model
        model.resize_token_embeddings(len(tokenizer))
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        if (
                args.model_name_or_path
                and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
        ):
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        if args.model_name_or_path and os.path.exists(args.model_name_or_path):
            try:
                checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0

        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
        )
        set_seed(args)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs, labels = (batch, batch)
                if inputs.shape[1] > 1024: continue
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                model.train()
                outputs = model(inputs, labels=labels)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        if (
                                args.local_rank == -1 and args.evaluate_during_training
                        ):
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        checkpoint_prefix = "checkpoint"
                        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step

    def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, df_trn, df_val, prefix="") -> Dict:
        eval_output_dir = args.output_dir

        eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True)
        os.makedirs(eval_output_dir, exist_ok=True)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, drop_last=True
        )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                outputs = model(inputs, labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return result

    args = Args(id)

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
            and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)

    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir,
    )
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, trn_df, val_df, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, trn_df, val_df, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


@app.route('/createChatbot', methods=['POST'])
def createChatbot():
    data = request.get_json()
    if 'id' in data:
        id = data['id']
        create_chatbot(id)
        return jsonify({'result': "success"})
    else:
        return jsonify({'error': 'ID not found in request'}), 400


if __name__ == '__main__':
    app.run(debug=True)
