"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
import random
import data_util
from data_util import ClothSample
import numpy as np
import torch
import time
from pytorch_pretrained_bert.modeling import BertForCloth
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import functools
def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
            
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='./data',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='cloth',
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='EXP/',
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--cache_size",
                        default=256,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_log_steps",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
        
    suffix = time.strftime('%Y%m%d-%H%M%S')
    args.output_dir = os.path.join(args.output_dir, suffix)
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging = get_logger(os.path.join(args.output_dir, 'log.txt'))
    
    data_file = {'train':'train',  'valid':'valid', 'test':'test'}
    for key in data_file.keys():
        data_file[key] = data_file[key] + '-' + args.bert_model + '.pt'
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logging("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logging("device {} n_gpu {} distributed training {}".format(device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    num_train_steps = None
    train_data = None
    if args.do_train:
        train_data = data_util.Loader(args.data_dir, data_file['train'], args.cache_size, args.train_batch_size, device)
        num_train_steps = int(
            train_data.data_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForCloth.from_pretrained(args.bert_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    
    global_step = 0
    if args.do_train:
        logging("***** Running training *****")
        logging("  Batch size = {}".format(args.train_batch_size))
        logging("  Num steps = {}".format(num_train_steps))

        model.train()
        for _ in range(int(args.num_train_epochs)):
            tr_loss = 0
            tr_acc = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for inp, tgt in train_data.data_iter():
                loss, acc, _, _ = model(inp, tgt)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    acc = acc.sum()
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                tr_acc += acc.item()
                #print(tr_acc)
                nb_tr_examples += inp[-2].sum()
                nb_tr_steps += 1
                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logging("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1
                if (global_step % args.num_log_steps == 0):
                    logging('step: {} | train loss: {} | train acc {}'.format(
                        global_step, tr_loss/nb_tr_examples, tr_acc/nb_tr_examples))
                    tr_loss = 0
                    tr_acc = 0
                    nb_tr_examples = 0

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logging("***** Running evaluation *****")
        logging("  Batch size = {}".format(args.eval_batch_size))
        valid_data = data_util.Loader(args.data_dir, data_file['valid'], args.cache_size, args.eval_batch_size, device)
        # Run prediction for full data

        model.eval()
        eval_loss, eval_accuracy, eval_h_acc, eval_m_acc = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, nb_eval_h_examples = 0, 0, 0
        for inp, tgt in valid_data.data_iter(shuffle=False):

            with torch.no_grad():
                tmp_eval_loss, tmp_eval_accuracy, tmp_h_acc, tmp_m_acc = model(inp, tgt)
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean() # mean() to average on multi-gpu.
                tmp_eval_accuracy = tmp_eval_accuracy.sum()
            eval_loss += tmp_eval_loss.item()
            eval_accuracy += tmp_eval_accuracy.item()
            eval_h_acc += tmp_h_acc.item()
            eval_m_acc += tmp_m_acc.item()
            nb_eval_examples += inp[-2].sum().item()
            nb_eval_h_examples += (inp[-2].sum(-1) * inp[-1]).sum().item()
            nb_eval_steps += 1         

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        eval_h_acc = eval_h_acc / nb_eval_h_examples
        eval_m_acc = eval_m_acc / (nb_eval_examples - nb_eval_h_examples)
        result = {'valid_eval_loss': eval_loss,
                  'valid_eval_accuracy': eval_accuracy,
                  'valid_h_acc':eval_h_acc,
                  'valid_m_acc':eval_m_acc,
                  'global_step': global_step}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logging("***** Valid Eval results *****")
            for key in sorted(result.keys()):
                logging("  {} = {}".format(key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))
                
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        test_data = data_util.Loader(args.data_dir, data_file['test'], args.cache_size, args.eval_batch_size, device)
        logging("***** Running test evaluation *****")
        logging("  Batch size = {}".format(args.eval_batch_size))
        # Run prediction for full data

        model.eval()
        eval_loss, eval_accuracy, eval_h_acc, eval_m_acc = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, nb_eval_h_examples = 0, 0, 0
        for inp, tgt in test_data.data_iter(shuffle=False):

            with torch.no_grad():
                tmp_eval_loss, tmp_eval_accuracy, tmp_h_acc, tmp_m_acc = model(inp, tgt)
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean() # mean() to average on multi-gpu.
                tmp_eval_accuracy = tmp_eval_accuracy.sum()
            eval_loss += tmp_eval_loss.item()
            eval_accuracy += tmp_eval_accuracy.item()
            eval_h_acc += tmp_h_acc.item()
            eval_m_acc += tmp_m_acc.item()
            nb_eval_examples += inp[-2].sum().item()
            nb_eval_h_examples += (inp[-2].sum(-1) * inp[-1]).sum().item()
            nb_eval_steps += 1         

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        eval_h_acc = eval_h_acc / nb_eval_h_examples
        eval_m_acc = eval_m_acc / (nb_eval_examples - nb_eval_h_examples)
        result = {'valid_eval_loss': eval_loss,
                  'valid_eval_accuracy': eval_accuracy,
                  'valid_h_acc':eval_h_acc,
                  'valid_m_acc':eval_m_acc,
                  'global_step': global_step}

        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logging("***** Test Eval results *****")
            for key in sorted(result.keys()):
                logging("  {} = {}".format(key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))
if __name__ == "__main__":
    main()
