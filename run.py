import os
import copy
from torchmetrics.functional import f1_score
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule
import torch
import ipdb

torch.cuda.empty_cache()


@ex.automain
def main(_config):  # variables from vilt.config.config().
    _config = copy.deepcopy(
        _config
    )  # this make it safe to modify variables in _config.
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)  # vilt.datamodules.multitask_datamodule

    model = ViLTransformerSS(
        _config
    )  # vilt.modules.vilt_missing_aware_prompt_module.VilTransformerSS
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,  # only the best one module would be saved.
        verbose=True,  # Whenever the model checkpoint is saved, relevant information will be output in the console.
        monitor="val/the_metric",
        mode="max",  # Save model when monitoring metrics increase.
        save_last=True,  # save the last module.
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],  # log_dir = "result"
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',  # [:-5] means delete '.ckpt'.
    )

    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval="step"
    )  # 监控学习率
    callbacks = [checkpoint_callback, lr_callback]
    # from pytorch_lightning.profiler import SimpleProfiler
    # profiler = SimpleProfiler()

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    # max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        max_steps=None,
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        # profiler=profiler,
    )

    if not _config["test_only"]:
        # ipdb.set_trace()
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
    # setup 方法在 trainer.fit 或 trainer.test 被调用


# hello, world! OK, it is connected!
