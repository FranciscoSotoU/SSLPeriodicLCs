from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import glob
import os
import numpy as np
import torch
os.environ['HYDRA_FULL_ERROR'] = '1'
torch.set_float32_matmul_precision('highest')
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.metrics import calculate_metrics
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    #model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        #"model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    #trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    classes = cfg.get('classes')
    baseline_f1s = cfg.get('baseline_f1s')
    # for predictions use trainer.predict(...)
    ckpt_path = cfg.ckpt_path
    ckpt_files = glob.glob(os.path.join(ckpt_path, "**", "*.ckpt"), recursive=True)

    ckpt_files_last = []
    ckpt_files_not_last = []
    for path in ckpt_files:
        if "last" in path:
            ckpt_files_last.append(path)
        else:
            ckpt_files_not_last.append(path)
    all_preds = []
    all_targets = []
    save_path = cfg.paths.output_dir
    for ckpt in ckpt_files_not_last:
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        out = trainer.predict(model=model, dataloaders=datamodule, ckpt_path=ckpt)
        new_out = []
        for o in out:
            new_out.append(o[0:2])
        out = np.concatenate(new_out, axis=1)
        preds = out[0]
        targets = out[1]
        all_preds.append(preds)
        all_targets.append(targets)
    
    calculate_metrics(all_targets,all_preds,save_path,name='best',classes=classes,compare_f1s=baseline_f1s)
    if len(ckpt_files_last) > 0:
        all_preds_last = []
        all_targets_last = []
        for ckpt in ckpt_files_last:
            model: LightningModule = hydra.utils.instantiate(cfg.model)
            out = trainer.predict(model=model, dataloaders=datamodule, ckpt_path=ckpt)
            new_out = []
            for o in out:
                new_out.append(o[0:2])
            out = np.concatenate(new_out, axis=1)
            preds = out[0]
            targets = out[1]
            all_preds_last.append(preds)
            all_targets_last.append(targets)
        calculate_metrics(all_preds_last,all_targets_last,save_path,name='last',classes=classes,compare_f1s=baseline_f1s)
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
