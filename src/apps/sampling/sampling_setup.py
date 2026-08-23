"""Setup helpers for sampling entrypoint."""

import torch

from src.data import data_module
from src.models.lightning.lightning_module import PlModel


def _override_covariate_embedding_paths(cov_cfg, runtime_cov_cfg):
    """Patch checkpoint covariate asset paths with runtime paths when provided."""
    if cov_cfg is None or runtime_cov_cfg is None:
        return cov_cfg
    for key in [
        "celltype_embedding_path",
        "gene_embedding_path",
        "pert_embedding_path",
        "drug_embedding_path",
        "replogle_gene_embedding_path",
    ]:
        val = runtime_cov_cfg.get(key, None)
        if val is not None:
            cov_cfg[key] = val
    return cov_cfg


def _apply_checkpoint_shape_patches(model, state_dict, logger):
    """Apply finetune-time projection replacement before strict state loading."""
    final_weight = state_dict.get("model.final_layer.linear.weight")
    if final_weight is None:
        return

    checkpoint_output_size = int(final_weight.shape[0])
    current_output_size = int(model.model.output_size)
    if checkpoint_output_size == current_output_size:
        return

    if getattr(model.model_cfg, "replace_2kgene_layer", False) or getattr(
        model.model_cfg, "replace_1w2gene_layer", False
    ):
        model.model.replace_2kgene_layer(new_input_size=checkpoint_output_size)
        logger.info(
            "Applied checkpoint projection replacement for sampling: %s -> %s genes",
            current_output_size,
            checkpoint_output_size,
        )
        return

    raise RuntimeError(
        "Checkpoint projection shape does not match initialized model "
        f"({checkpoint_output_size} vs {current_output_size}) and no replacement flag is set."
    )


def build_sampling_datamodule(cfg, logger):
    """
    Build sampling datamodule.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :return: Requested object(s) for downstream use.
    """
    if cfg.data.data_name in ["PBMCFinetune"]:
        datamodule = data_module.PBMCPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.sampling.batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    elif cfg.data.data_name in ["Tahoe100mFinetune", "ReplogleFinetune"]:
        datamodule = data_module.Tahoe100mPerturbationDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    elif cfg.data.data_name in [
        "Tahoe100mPBMCPretrain",
        "CellxGenePretrain",
        "PBMCReploglePretrain",
        "Tahoe100mPBMCReplogleCellxGenePretrain",
        "Tahoe100mPBMCCellxGenePretrain",
        "PBMCReplogleCellxGenePretrain",
        "ReplogleCellxGenePretrain",
        "Tahoe100mCellxGenePretrain",
    ]:
        datamodule = data_module.PerturbationPretrainingDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.optimization.micro_batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    else:
        assert not cfg.data.data_name.endswith("Finetune")
        datamodule = data_module.PretrainingDataModule(
            seed=cfg.optimization.seed,
            micro_batch_size=cfg.sampling.batch_size,
            data_args=cfg.data,
            py_logger=logger,
        )
    datamodule.replace_pert_dict = cfg.cov_encoding.get("replace_pert_dict", False)
    datamodule.setup()
    return datamodule


def populate_covariate_cfg(cfg, datamodule):
    """Execute `populate_covariate_cfg` and return values used by downstream logic."""
    cfg.cov_encoding.num_pert = len(datamodule.pert_dict)
    cfg.cov_encoding.num_celltype = len(datamodule.cell_type_dict)
    cfg.cov_encoding.num_batch = len(datamodule.batch_dict)

    cfg.cov_encoding.pert_dict = datamodule.pert_dict
    cfg.cov_encoding.cell_type_dict = datamodule.cell_type_dict
    cfg.cov_encoding.batch_dict = datamodule.batch_dict
    cfg.model.dataset_dict = datamodule.original_dataset_name_list


def load_sampling_model(cfg, logger, datamodule):
    """
    Load sampling model.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :param datamodule: Data module providing datasets and loaders.
    :return: Requested object(s) for downstream use.
    """
    ckpt = torch.load(cfg.model_checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    #needs_cov = "cov_encoding_cfg" in hparams
    #needs_model = "model_cfg" in hparams
    #needs_opt = "optimizer_cfg" in hparams
    
    # using the training setting
    needs_cov = needs_model = needs_opt = True
    cov_cfg = hparams["cov_encoding_cfg"] if needs_cov else cfg.cov_encoding
    model_cfg = hparams["model_cfg"] if needs_model else cfg.model
    optimizer_cfg = hparams["optimizer_cfg"] if needs_opt else cfg.optimization

    cov_cfg = _override_covariate_embedding_paths(cov_cfg, cfg.cov_encoding)
    cov_cfg["celltype_encoding"] = cfg.cov_encoding.celltype_encoding

    model = PlModel(
        cov_encoding_cfg=cov_cfg,
        model_cfg=model_cfg,
        optimizer_cfg=optimizer_cfg,
        py_logger=logger,
        trainer_cfg=cfg.trainer,
        all_split_names=datamodule.all_split_names,
    )
    _apply_checkpoint_shape_patches(model, ckpt["state_dict"], logger)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model
