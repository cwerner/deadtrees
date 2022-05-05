import glob
import os

import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeToWandbAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(
                os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True
            ):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


# source: https://github.com/PyTorchLightning/pytorch-lightning/discussions/9910
class LogConfusionMatrixToWandbVal(Callback):
    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

        # def on_validation_batch_end(
        #         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        # ):
        #     if self.ready:
        #         self.preds.append(outputs[OutputKeys.PREDICTION].detach().cpu().numpy())
        #         self.targets.append(outputs[OutputKeys.TARGET].detach().cpu().numpy())

        # #@rank_zero_only
        # def on_validation_epoch_end(self, trainer, pl_module):
        #     if not self.ready:
        #         return

        #     logger = get_wandb_logger(trainer)
        #     experiment = logger.experiment

        #     experiment.log(
        #         {
        #             "conf_mat": wandb.plot.confusion_matrix(
        #                 probs=None,
        #                 y_true=target,
        #                 preds=prediction,
        #                 class_names=["BG", "NEEDLELEAF", "BROADLEAF"],
        #             )
        #         }
        #     )
        # conf_mat_name = f'CM_epoch_{trainer.current_epoch}'
        # logger = get_wandb_logger(trainer)
        # experiment = logger.experiment

        # preds = []
        # for step_pred in self.preds:
        #     preds.append(trainer.model.module.module.to_metrics_format(np.array(step_pred)))

        # preds = np.concatenate(preds).flatten()
        # targets = np.concatenate(np.array(self.targets)).flatten()

        # num_classes = max(np.max(preds), np.max(targets)) + 1

        # conf_mat = confusion_matrix(
        #     target=torch.tensor(targets),
        #     preds=torch.tensor(preds),
        #     num_classes=num_classes
        #     )

        # # set figure size
        # plt.figure(figsize=(14, 8))
        # # set labels size
        # sn.set(font_scale=1.4)
        # # set font size
        # fig = sn.heatmap(conf_mat, annot=True, annot_kws={"size": 8}, fmt="g")

        # for i in range(conf_mat.shape[0]):
        #     fig.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=3))
        # plt.xlabel('Predictions')
        # plt.ylabel('Targets')
        # plt.title(conf_mat_name)

        # conf_mat_path = Path(os.getcwd()) / 'conf_mats' / 'val'
        # conf_mat_path.mkdir(parents=True, exist_ok=True)
        # conf_mat_file_path = conf_mat_path / (conf_mat_name + '.txt')
        # df = pd.DataFrame(conf_mat.detach().cpu().numpy())

        # # save as csv or tsv to disc
        # df.to_csv(path_or_buf=conf_mat_file_path, sep='\t')
        # # save tsv to wandb
        # experiment.save(glob_str=str(conf_mat_file_path), base_path=os.getcwd())
        # # names should be uniqe or else charts from different experiments in wandb will overlap
        # experiment.log({f"confusion_matrix_val_img/ep_{trainer.current_epoch}": wandb.Image(plt)},
        #                commit=False)
        # # according to wandb docs this should also work but it crashes
        # # experiment.log(f{"confusion_matrix/{experiment.name}": plt})
        # # reset plot
        # plt.clf()
        self.preds.clear()
        self.targets.clear()
