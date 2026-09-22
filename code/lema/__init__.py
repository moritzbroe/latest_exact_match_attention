"""LEMA: latest exact match attention, with its training recipe and an off-GPU
retrieval store for inference.

    lema.train.train(model_cfg, task, train_cfg)   the hardening recipe (see train.py)
    lema.model.load_trained(run_dir)               a finished run, ready to evaluate
    lema.analysis.trace(model, x)                  what every layer attends to
    lema.cache / lema.decode                       the hash-table cache and the decoders
"""
from .model import ModelConfig, Transformer, best_dtype, load_trained
from .tasks import Recall, TokenCorpus
from .train import TrainConfig, train
