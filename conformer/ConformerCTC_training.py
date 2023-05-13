# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2-dev
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

import nemo.collections.asr as nemo_asr

# model = nemo_asr.models.ASRModel.from_pretrained(model_name='stt_de_conformer_ctc_large')
model = nemo_asr.models.ASRModel.from_pretrained(model_name='stt_de_conformer_ctc_large')
model.cfg

train_manifest = "/home/paperspace/conformer/dataset_train/manifest.json"

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import datetime
from pathlib import Path
import copy
from pytorch_lightning.loggers import WandbLogger  
import wandb
from ruamel.yaml import YAML
import functools
import copy
import pytorch_lightning as pl

def sweep_iteration():
    trainer = pl.Trainer(max_epochs=10)
    
    # setup model - note how we refer to sweep parameters with wandb.config
    model = nemo_asr.models.ASRModel.from_pretrained(model_name='stt_de_conformer_ctc_large')

    model.set_trainer(trainer)
    
    model.cfg.train_ds.is_tarred = False
    
    model.cfg.train_ds.manifest_filepath = train_manifest
    model.cfg.validation_ds.manifest_filepath = str(train_manifest)
    model.cfg.test_ds.manifest_filepath = str(train_manifest)
   
    
    model.cfg.train_ds.max_duration = 45
    model.cfg.train_ds.batch_size = 32
    model.cfg.validation_ds.batch_size = 32
    model.cfg.test_ds.batch_size = 32
    
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_test_data(model.cfg.test_ds)
    model.setup_optimization(model.cfg.optim)

    # train
    trainer.fit(model)

sweep_iteration()
