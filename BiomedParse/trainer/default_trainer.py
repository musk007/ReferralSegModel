# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import time
import os
import sys
import importlib
import json
import random
import wandb
import logging
import numpy as np
import copy
import contextlib
import shutil
from typing import Any, Callable, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI
from infinibatch import iterators

from .distributed_trainer import DistributedTrainer
from .utils_trainer import UtilsTrainer
from .utils.misc import *
from .utils.serialization import JSONEncoder, filter_jsonable

logger = logging.getLogger(__name__)


def _flatten_eval_results_for_wandb(results: dict, prefix: str = "eval") -> dict:
    """
    Flatten nested eval results into wandb-friendly scalar dict.
    Only includes scalar values (int, float); skips lists/dicts (e.g. instance_results).
    """
    flat = {}
    for key, val in results.items():
        if isinstance(val, (int, float)):
            flat[f"{prefix}/{key}"] = float(val)
        elif isinstance(val, dict):
            sub = _flatten_eval_results_for_wandb(val, prefix=f"{prefix}/{key}")
            flat.update(sub)
        # skip lists, arrays, etc.
    return flat


class DefaultTrainer(UtilsTrainer, DistributedTrainer):

    def __init__(self, opt):
        """
        Set up the task the model is being trained for.
        """
        super().__init__(opt)
        base_name = 'base_dir'
        base_path =  os.path.join(self.opt['base_path'], '__init__.py')
        spec = importlib.util.spec_from_file_location(base_name, base_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[base_name] = module
        spec.loader.exec_module(module)
        logger.info(f"Imported {base_name} at base_path {self.opt['base_path']}")

        pipeline_module = importlib.import_module(f"base_dir.pipeline.{self.opt['PIPELINE']}")
        pipeline_class = getattr(pipeline_module, self.opt['PIPELINE'])
        logger.info(f"Pipeline for training: {self.opt['PIPELINE']}")
        self.pipeline = pipeline_class(self.opt)

    def eval(self, ):
        logger.info('-----------------------------------------------')
        logger.info("Evaluating model ... ")
        self.mode = "eval"

        # self.model_names, self.raw_models, self.criteria = self.pipeline.set_up_model()
        self.raw_models = self.pipeline.initialize_model()
        self.model_names = self.raw_models.keys()

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.opt['device'])

        # load model during evaluation
        resume_from = self.opt.get('RESUME_FROM', '')
        if self.opt.get('WEIGHT') and resume_from and (
            os.path.isfile(resume_from) or os.path.isdir(resume_from) or
            (isinstance(resume_from, str) and resume_from.startswith('hf_hub:'))
        ):
            self.load_model(resume_from)
        else:
            raise ValueError(f"Model not found: {resume_from!r}")

        results = self._eval_on_set(self.save_folder)
        return results

    def _eval_on_set(self, save_folder):
        logger.info(f"Evaluation start ...")
        if self.opt['FP16']:
            from torch.cuda.amp import autocast
            with autocast():
                results = self.pipeline.evaluate_model(self, save_folder)
        else:        
            results = self.pipeline.evaluate_model(self, save_folder)
        if self.opt['rank'] == 0:
            logger.info(results)
        # Barrier: wait for all ranks to finish evaluation before resuming training.
        # Without this, rank 0 can race ahead into the next epoch's ALLREDUCE
        # while other ranks are still inside the evaluator's all_gather, causing
        # an NCCL collective mismatch and a 600-second watchdog timeout/crash.
        if self.opt['world_size'] > 1:
            torch.distributed.barrier()
        return results

    def _eval_on_eval_split(self, save_folder):
        """Evaluate on DATASETS.EVAL (the held-out validation split)."""
        eval_datasets = self.opt.get('DATASETS', {}).get('EVAL', [])
        if not eval_datasets:
            return {}
        logger.info("Eval-split evaluation start ...")
        if self.opt['FP16']:
            from torch.cuda.amp import autocast
            with autocast():
                results = self.pipeline.evaluate_model_on_datasets(self, save_folder, eval_datasets)
        else:
            results = self.pipeline.evaluate_model_on_datasets(self, save_folder, eval_datasets)
        # Barrier: same reason as _eval_on_set — prevent rank divergence after evaluation.
        if self.opt['world_size'] > 1:
            torch.distributed.barrier()
        return results

    @staticmethod
    def _extract_metric(results: dict, metric_key: str) -> float:
        """
        Walk the nested results dict and return the mean of all values matching
        `metric_key`.  Results structure:
            { "<dataset>/<eval_type>": { "<subtype>": { "<metric>": value, ... } } }
        Returns -inf if the key is not found.
        """
        values = []
        def _walk(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == metric_key and isinstance(v, (int, float)):
                        values.append(float(v))
                    else:
                        _walk(v)
        _walk(results)
        return float(np.mean(values)) if values else float('-inf')

    @staticmethod
    def _summarise_eval_metrics(results: dict) -> dict:
        """
        Flatten the nested eval results dict into a single-level dict of
        mean values for the key segmentation metrics (mIoU, mDice, cIoU, cDice).
        Returns { metric_name: mean_value } across all datasets in results.
        """
        REPORT_KEYS = ("mIoU", "mDice", "cIoU", "cDice", "precision@0.5")
        accum = {k: [] for k in REPORT_KEYS}

        def _walk(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k in REPORT_KEYS and isinstance(v, (int, float)):
                        accum[k].append(float(v))
                    else:
                        _walk(v)
        _walk(results)
        return {k: float(np.mean(v)) for k, v in accum.items() if v}

    def compute_loss(self, forward_func, batch):

        def forward(func, trainer, batch):
            if self.opt['FP16']:
                from torch.cuda.amp import autocast
                with autocast():
                    loss = func(trainer, batch)
            else:
                loss = func(trainer, batch)
            return loss

        loss = forward(forward_func, self, batch)
        return loss

    def backward_loss(self, loss, model_names=['default']):  # noqa: E252

        def backward(loss_tensor):
            if self.opt['FP16']:
                self.grad_scaler.scale(loss_tensor).backward()
            else:
                loss_tensor.backward()
            
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps

        backward(loss)
        return loss

    def update_model(self, model_name='default'):
        if self.opt['FP16']:
            self.grad_scaler.unscale_(self.optimizers[model_name])
            self.grad_scaler.step(self.optimizers[model_name])
        else:
            self.optimizers[model_name].step()

        self.optimizers[model_name].zero_grad()
        self.train_params['optim_steps'][model_name] += 1
        self.lr_schedulers[model_name].step()

    def train_step(self, batch):
        self.grad_acc_batches.append(batch) # support batch accumulation

        if self.is_gradient_accumulation_boundary():
            # set all modules and criteria into training mode
            for model_name in self.model_names:
                self.models[model_name].train()

            assert len(self.grad_acc_batches) == self.grad_acc_steps

            total_batch_sample = 0
            for batch_index, batch in enumerate(self.grad_acc_batches):

                loss_info, sample_size_info, extra_info = \
                    self.pipeline.forward_step(self,
                                            batch,
                                            self.grad_acc_batches,
                                            batch_index,
                                            is_distributed=(self.opt['world_size'] > 1))

                self.train_loss.update_iter(loss_info)
                total_batch_sample += sample_size_info['num_samples']

            if self.opt['FP16']:
                # Update GradScaler after an effective batch
                self.grad_scaler.update()

            # update losses and item counts of an effective batch to the AverageMeters
            if self.opt['world_size'] > 1:
                total_batch_sample = torch.tensor(total_batch_sample).to(self.opt['device'])
                torch.distributed.all_reduce(total_batch_sample, torch.distributed.ReduceOp.SUM)
                total_batch_sample = total_batch_sample.item()

            self.train_params['total_batch_size'] += total_batch_sample
            self.grad_acc_batches = []

        self.train_params['num_updates'] += 1
        
    def init_train(self):
        self.mode = "train"
        logger.info('-------------------------------------------------------')
        logger.info("Training on rank: {}".format(self.opt['rank']))

        self.raw_models = self.pipeline.initialize_model()
        self.model_names = list(self.raw_models.keys())

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.opt['device'])

        self.train_dataloaders = self.pipeline.get_dataloaders(self, 'train', is_evaluation=False)

        try:
            _updates_per_epoch = len(self.train_dataloaders)
        except TypeError:
            # TrainingSampler is infinite and has no __len__.
            # Derive epoch length from total dataset samples / batch size.
            import math
            from detectron2.data import DatasetCatalog
            _total_samples = sum(
                len(DatasetCatalog.get(name))
                for name in self.opt['DATASETS']['TRAIN']
            )
            _batch_size = self.opt['TRAIN']['BATCH_SIZE_TOTAL']
            _updates_per_epoch = math.ceil(_total_samples / _batch_size)
            logging.getLogger(__name__).info(
                f"Infinite sampler detected: computed updates_per_epoch="
                f"{_updates_per_epoch} from {_total_samples} samples / batch {_batch_size}"
            )

        self.train_params = {
                             "updates_per_epoch": _updates_per_epoch,
                             "total_batch_size": 0,
                             "num_updates": 0,
                             "optim_steps": {module_name: 0 for module_name in self.model_names},
                             "start_epoch_idx": 0,
                             "start_batch_idx": 0,
                             "current_epoch_idx": 0,
                             "current_batch_idx": 0,
                             "resume_epoch_idx": 0, 
                             }

        self.train_loss = LossMeter()
        self.grad_acc_batches = []

        if self.opt['CUDA']:
            torch.cuda.empty_cache()

        self.create_optimizer_and_scheduler()
        self.models = {model_name: self.raw_models[model_name] for model_name in self.model_names}
        self._initialize_ddp()

        if self.opt.get('WEIGHT', False):
            self.load_weight(self.opt['RESUME_FROM'], must_exist=True)
        if self.opt.get('RESUME', False):
            self.load_checkpoint(self.opt['RESUME_FROM'], must_exist=True)

        ######################
        # Start the main loop
        ######################
        if self.opt['rank'] == 0:
            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num of GPUs = {self.opt['world_size']}")
            logger.info(f"  Num Epochs = {self.opt['SOLVER']['MAX_NUM_EPOCHS']}")
            logger.info(f"  Num of Mini Batches per Epoch = {self.train_params['updates_per_epoch']}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.opt['SOLVER']['MAX_NUM_EPOCHS'] * self.train_params['updates_per_epoch']}")
            logger.info(f"  Gradient Accumulation steps = {self.grad_acc_steps}")
            logger.info(f"  Total optimization steps = {self.opt['SOLVER']['MAX_NUM_EPOCHS'] * self.train_params['updates_per_epoch'] // self.grad_acc_steps}")

            # Initialize wandb on rank 0 (must happen before any wandb.log in the training loop)
            if self.opt.get('WANDB', False):
                if 'WANDB_KEY' not in os.environ:
                    logger.warning("WANDB_KEY not set; wandb logging disabled. Set export WANDB_KEY=... in runner script.")
                else:
                    try:
                        wandb.login(key=os.environ['WANDB_KEY'])
                        wandb_dir = os.path.join(self.save_folder, 'wandb')
                        os.makedirs(wandb_dir, exist_ok=True)
                        runid = None
                        if os.path.exists(os.path.join(wandb_dir, 'runid.txt')):
                            runid = open(os.path.join(wandb_dir, 'runid.txt')).read()
                        init_kwargs = dict(
                            project=self.opt.get('WANDB_PROJECT', 'BiomedParseFineTune'),
                            name=str(self.save_folder),
                            dir=wandb_dir,
                            resume="allow",
                            id=runid,
                        )
                        if self.opt.get('WANDB_ENTITY'):
                            init_kwargs["entity"] = self.opt['WANDB_ENTITY']
                        wandb.init(**init_kwargs)
                        open(os.path.join(wandb_dir, 'runid.txt'), 'w').write(wandb.run.id)
                        logger.info("wandb initialized successfully (run: %s)", getattr(wandb.run, 'url', wandb.run.id))
                    except Exception as e:
                        logger.warning("wandb init failed: %s", e)

    def train(self):
        """
        Training
        """
        self.init_train()
        current_optim_steps = self._get_and_validate_current_optim_steps()
        num_epochs = self.opt['SOLVER']['MAX_NUM_EPOCHS']

        if self.opt.get('EVAL_AT_START', False):
            results = self._eval_on_set(self.save_folder)
            if self.opt['rank'] == 0 and self.opt.get('WANDB', False) and wandb.run is not None:
                eval_log = _flatten_eval_results_for_wandb(results)
                if eval_log:
                    wandb.log(eval_log, step=0)

        best_eval_score = float('-inf')
        best_metric_key = self.opt.get('SOLVER', {}).get('BEST_METRIC', 'mIoU')
        eval_datasets = self.opt.get('DATASETS', {}).get('EVAL', [])
        if eval_datasets and self.opt['rank'] == 0:
            logger.info(f"Eval-split datasets: {eval_datasets}  (tracking metric: {best_metric_key})")

        train_prev_logged_time = datetime.now()
        for epoch in range(self.train_params['start_epoch_idx'], num_epochs):
            self.train_params['current_epoch_idx'] = epoch
            logger.info(f"Start epoch: {epoch} training.")
            
            epoch_start_time = datetime.now()
            for batch_idx, batch in enumerate(self.train_dataloaders):
                if self.train_params['current_epoch_idx'] == self.train_params['start_epoch_idx']:
                    if batch_idx < self.train_params['start_batch_idx']: # skip the first few batches for resuming
                        continue

                self.train_params['current_batch_idx'] = batch_idx
                prev_optim_steps = current_optim_steps
                prev_total_batch_size = self.train_params['total_batch_size']

                # update
                self.prev_optim_steps = prev_optim_steps
                self.train_step(batch)

                current_optim_steps = self._get_and_validate_current_optim_steps()
                
                # logging
                if prev_optim_steps != current_optim_steps:  # an optimizer update was made
                    log_first = self.opt.get("LOG_FIRST", 10)
                    log_every = self.opt.get("LOG_EVERY", 100)
                    if (current_optim_steps % log_every == 0) or (epoch == 0 and current_optim_steps <= log_first): # print logging

                        last_lr = {}
                        for module_name in self.model_names:
                            last_lr[module_name] = self.lr_schedulers[module_name].get_last_lr()[0]

                        train_time_delta = (datetime.now() - train_prev_logged_time).total_seconds()
                        train_prev_logged_time = datetime.now()
                        MB = 1024.0 * 1024.0
                        memory = torch.cuda.max_memory_allocated() / MB

                        if self.opt['rank'] == 0:
                            total_loss = sum(obj.val for obj in self.train_loss.losses.values())
                            total_loss_avg = sum(obj.avg for obj in self.train_loss.losses.values())
                            items_this_log = self.train_params['total_batch_size'] - prev_total_batch_size
                            its = items_this_log / train_time_delta
                            eta = str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (self.train_params['updates_per_epoch'] - batch_idx - 1)).split('.')[0]
                            lr_str = ', '.join([f'{k}: {v:.2e}' for k, v in last_lr.items()])
                            logger.info(
                                f"epoch[{epoch+1:3}/{self.opt['SOLVER']['MAX_NUM_EPOCHS']}]"
                                f"  step[{batch_idx+1:4}/{self.train_params['updates_per_epoch']}]"
                                f"  loss[{total_loss:.4f} / avg {total_loss_avg:.4f}]"
                                f"  lr[{lr_str}]"
                                f"  {its:.1f} img/s"
                                f"  mem[{memory:.0f} MB]"
                                f"  eta[{eta}]"
                            )
                            if self.opt.get('WANDB', False) and wandb.run is not None:
                                log_dict = {"train/loss": total_loss, "train/loss_avg": total_loss_avg, "train/img_per_sec": its, "train/mem_mb": memory}
                                for k, v in last_lr.items():
                                    log_dict[f"train/lr_{k}"] = v
                                wandb.log(log_dict, step=current_optim_steps)

                # evaluate and save ckpt every epoch
                if batch_idx + 1 == self.train_params['updates_per_epoch']:
                    if self.opt.get('SAVE_CHECKPOINT', True):
                        self.save_checkpoint(self.train_params['num_updates'])

                    # --- Test-set evaluation (unchanged) ---
                    results = self._eval_on_set(self.save_folder)
                    if self.opt['rank'] == 0 and self.opt.get('WANDB', False) and wandb.run is not None:
                        eval_log = _flatten_eval_results_for_wandb(results, prefix="test")
                        if eval_log:
                            wandb.log(eval_log, step=current_optim_steps)

                    # --- Eval-split evaluation + best-model tracking ---
                    if eval_datasets:
                        eval_results = self._eval_on_eval_split(self.save_folder)
                        score = self._extract_metric(eval_results, best_metric_key)
                        summary = self._summarise_eval_metrics(eval_results)
                        if self.opt['rank'] == 0:
                            metrics_str = "  ".join(
                                f"{k}={v:.4f}" for k, v in summary.items()
                            )
                            best_marker = "  *** NEW BEST ***" if score > best_eval_score else \
                                          f"  (best: {best_eval_score:.4f})"
                            logger.info(
                                f"EVAL epoch[{epoch+1:3}/{num_epochs}]  "
                                + metrics_str
                                + best_marker
                            )
                            if self.opt.get('WANDB', False) and wandb.run is not None:
                                eval_log = {f"eval/{k}": v for k, v in summary.items()}
                                eval_log[f"eval/best_{best_metric_key}"] = max(score, best_eval_score)
                                eval_log["epoch"] = epoch + 1
                                wandb.log(eval_log, step=current_optim_steps)
                                # also update summary so metrics are always visible in workspace table
                                for k, v in summary.items():
                                    wandb.run.summary[f"eval/{k}"] = v
                                wandb.run.summary[f"eval/best_{best_metric_key}"] = max(score, best_eval_score)
                                wandb.run.summary["epoch"] = epoch + 1
                        if score > best_eval_score:
                            best_eval_score = score
                            self.save_best_checkpoint(epoch, score, best_metric_key)

                    # Synchronise ALL ranks at the very end of the epoch (after
                    # checkpoint saving, test-set eval, and eval-split eval).
                    # This prevents rank 0 from racing ahead or blocking at a
                    # rank-0-only operation (e.g. save_best_checkpoint) while
                    # other ranks have already started the next epoch's training.
                    if self.opt['world_size'] > 1:
                        torch.distributed.barrier()
                    break

            logger.info(f"This epoch takes {datetime.now() - epoch_start_time}")
            logger.info(f"PROGRESS: {100.0 * (epoch + 1) / num_epochs:.2f}%")
            logger.info(f"Config files are at {self.opt['conf_files']}")

        # if not self.opt.get('SAVE_CHECKPOINT', True):
        #     self.save_checkpoint(self.train_params['num_updates'])