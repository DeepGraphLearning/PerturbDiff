"""Inference-only API for sampling perturbation responses from control AnnData."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import anndata
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy import sparse

from src.apps.sampling.sampling_setup import (
    _apply_checkpoint_shape_patches,
    _override_covariate_embedding_paths,
)
from src.models.lightning.lightning_module import PlModel


LOGGER = logging.getLogger(__name__)


def _first_existing_column(adata: anndata.AnnData, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in adata.var:
            return col
    return None


def _as_dense_float32(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _normalize_name_keys(mapping: Dict) -> Dict[str, int]:
    return {str(k): int(v) for k, v in mapping.items()}


class PerturbDiffPredictor:
    """Load a PerturbDiff checkpoint once and run h5ad-based inference."""

    def __init__(
        self,
        checkpoint_path: str,
        selected_gene_file: str,
        device: str = "auto",
        covariate_asset_paths: Optional[Dict[str, object]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.selected_gene_file = str(selected_gene_file)
        self.logger = logger or LOGGER
        self.device = self._resolve_device(device)

        with open(self.selected_gene_file, "rb") as fin:
            self.genes = list(pickle.load(fin))

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        hparams = ckpt["hyper_parameters"]
        cov_cfg = OmegaConf.create(hparams["cov_encoding_cfg"])
        model_cfg = OmegaConf.create(hparams["model_cfg"])
        optimizer_cfg = OmegaConf.create(hparams["optimizer_cfg"])
        trainer_cfg = OmegaConf.create(hparams["trainer_cfg"])

        if covariate_asset_paths:
            cov_cfg = _override_covariate_embedding_paths(cov_cfg, covariate_asset_paths)

        self.model = PlModel(
            cov_encoding_cfg=cov_cfg,
            model_cfg=model_cfg,
            optimizer_cfg=optimizer_cfg,
            py_logger=self.logger,
            trainer_cfg=trainer_cfg,
            all_split_names=["test"],
        )
        _apply_checkpoint_shape_patches(self.model, ckpt["state_dict"], self.logger)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.cov_cfg = cov_cfg
        self.model_cfg = model_cfg
        self.pert_dict = _normalize_name_keys(cov_cfg.pert_dict)
        self.cell_type_dict = _normalize_name_keys(cov_cfg.cell_type_dict)
        self.batch_dict = _normalize_name_keys(cov_cfg.batch_dict)

        self.supported_dataset_names = list(model_cfg.dataset_dict)
        self.default_dataset_name = self._first_dataset_name_containing("tahoe")
        self.default_batch_key = self._first_batch_for_dataset(self.default_dataset_name)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _first_dataset_name_containing(self, pattern: str) -> str:
        for name in self.supported_dataset_names:
            if pattern.lower() in str(name).lower():
                return str(name)
        return str(self.supported_dataset_names[0])

    def _first_batch_for_dataset(self, dataset_name: str) -> str:
        prefix = f"{dataset_name}_"
        for key in self.batch_dict:
            if key.startswith(prefix):
                return key
        return next(iter(self.batch_dict))

    def list_perturbations(self, contains: Optional[str] = None, limit: int = 20) -> List[str]:
        vals = sorted(self.pert_dict)
        if contains:
            vals = [v for v in vals if contains.lower() in v.lower()]
        return vals[:limit]

    def list_cell_types(self, contains: Optional[str] = None, limit: int = 20) -> List[str]:
        vals = sorted(self.cell_type_dict)
        if contains:
            vals = [v for v in vals if contains.lower() in v.lower()]
        return vals[:limit]

    def validate_request(self, perturbation: str, cell_type: str, batch_key: Optional[str] = None) -> None:
        missing = []
        if perturbation not in self.pert_dict:
            missing.append(f"perturbation={perturbation!r}")
        if cell_type not in self.cell_type_dict:
            missing.append(f"cell_type={cell_type!r}")
        if batch_key is not None and batch_key not in self.batch_dict:
            missing.append(f"batch_key={batch_key!r}")
        if missing:
            raise ValueError(
                "Unsupported checkpoint covariate(s): "
                + ", ".join(missing)
                + ". Use list_perturbations()/list_cell_types() or inspect batch_dict for supported values."
            )

    def align_adata(
        self,
        adata: anndata.AnnData,
        gene_column: Optional[str] = None,
        layer: Optional[str] = None,
        min_gene_overlap: float = 0.8,
    ) -> np.ndarray:
        gene_names = self._resolve_gene_names(adata, gene_column)
        gene_to_idx = {str(g): i for i, g in enumerate(gene_names)}
        column_map = np.array([gene_to_idx.get(str(g), -1) for g in self.genes], dtype=np.int64)
        overlap = float((column_map >= 0).mean())
        if overlap < min_gene_overlap:
            raise ValueError(
                f"Only {overlap:.1%} of checkpoint genes were found in the input AnnData; "
                f"required at least {min_gene_overlap:.1%}."
            )

        source = adata.layers[layer] if layer else adata.X
        aligned = np.zeros((adata.n_obs, len(self.genes)), dtype=np.float32)
        existing = column_map >= 0
        aligned[:, existing] = _as_dense_float32(source[:, column_map[existing]])
        return aligned

    def _resolve_gene_names(self, adata: anndata.AnnData, gene_column: Optional[str]) -> List[str]:
        if gene_column is None:
            gene_column = _first_existing_column(
                adata,
                ["feature_name", "gene_name", "gene_symbols", "symbol", "gene_symbol"],
            )
        if gene_column is None:
            return [str(x) for x in adata.var_names]
        return [str(x) for x in adata.var[gene_column].tolist()]

    @torch.no_grad()
    def predict_adata(
        self,
        adata: anndata.AnnData,
        perturbation: str,
        cell_type: str,
        batch_key: Optional[str] = None,
        dataset_name: Optional[str] = None,
        gene_column: Optional[str] = None,
        layer: Optional[str] = None,
        normalize_counts: Optional[float] = 10.0,
        batch_size: int = 128,
        start_time: int = 100,
        use_ddim: bool = True,
        eta: float = 0.0,
        guidance_strength: float = 1.0,
        clip_denoised: bool = True,
        progress: bool = False,
        min_gene_overlap: float = 0.8,
    ) -> anndata.AnnData:
        """Predict perturbed expression for each input control cell."""
        batch_key = batch_key or self.default_batch_key
        dataset_name = dataset_name or self.default_dataset_name
        self.validate_request(perturbation, cell_type, batch_key)

        x_ctrl = self.align_adata(adata, gene_column=gene_column, layer=layer, min_gene_overlap=min_gene_overlap)
        if normalize_counts:
            x_ctrl = x_ctrl / float(normalize_counts)

        outputs = []
        for start in range(0, x_ctrl.shape[0], batch_size):
            x_chunk = x_ctrl[start : start + batch_size]
            outputs.append(
                self._predict_array_chunk(
                    x_chunk,
                    perturbation=perturbation,
                    cell_type=cell_type,
                    batch_key=batch_key,
                    dataset_name=dataset_name,
                    start_time=start_time,
                    use_ddim=use_ddim,
                    eta=eta,
                    guidance_strength=guidance_strength,
                    clip_denoised=clip_denoised,
                    progress=progress,
                )
            )

        pred = np.concatenate(outputs, axis=0)
        if normalize_counts:
            pred = pred * float(normalize_counts)

        obs = adata.obs.copy()
        obs["perturbdiff_perturbation"] = perturbation
        obs["perturbdiff_cell_type"] = cell_type
        obs["perturbdiff_batch_key"] = batch_key
        obs["perturbdiff_dataset_name"] = dataset_name
        return anndata.AnnData(X=pred, obs=obs, var=pd.DataFrame(index=pd.Index(self.genes, name="gene")))

    def predict_file(self, input_h5ad: str, output_h5ad: str, **kwargs) -> anndata.AnnData:
        adata = anndata.read_h5ad(input_h5ad)
        pred = self.predict_adata(adata, **kwargs)
        Path(output_h5ad).parent.mkdir(parents=True, exist_ok=True)
        pred.write_h5ad(output_h5ad)
        return pred

    def _predict_array_chunk(
        self,
        x_ctrl: np.ndarray,
        perturbation: str,
        cell_type: str,
        batch_key: str,
        dataset_name: str,
        start_time: int,
        use_ddim: bool,
        eta: float,
        guidance_strength: float,
        clip_denoised: bool,
        progress: bool,
    ) -> np.ndarray:
        bsz = x_ctrl.shape[0]
        cont_emb = torch.as_tensor(x_ctrl, dtype=torch.float32, device=self.device).unsqueeze(1)
        cov_pert = torch.full((bsz, 1), self.pert_dict[perturbation], dtype=torch.long, device=self.device)
        cov_celltype = torch.full((bsz, 1), self.cell_type_dict[cell_type], dtype=torch.long, device=self.device)
        cov_batch = torch.full((bsz, 1), self.batch_dict[batch_key], dtype=torch.long, device=self.device)
        batch = {
            "cont_emb": cont_emb,
            "cov_pert": cov_pert,
            "cov_celltype": cov_celltype,
            "cov_batch": cov_batch,
        }
        batch["batch_emb"] = self.model._encode_covariates(batch)
        self_condition = {
            "batch_emb": batch["batch_emb"],
            "cont_emb": cont_emb,
            "gene_emb": None,
            "ds_name": [dataset_name] * bsz,
        }

        sample_fn = self.model.diffusion.ddim_sample_loop if use_ddim else self.model.diffusion.p_sample_loop
        kwargs = {
            "self_condition": self_condition,
            "clip_denoised": clip_denoised,
            "device": self.device,
            "progress": progress,
            "start_time": min(int(start_time), int(self.model_cfg.steps)),
            "guidance_strength": guidance_strength,
            "sample_kwargs": {},
        }
        if use_ddim:
            kwargs["eta"] = eta
        else:
            kwargs["nw"] = 0.5
            kwargs["start_guide_steps"] = 500

        sample, _ = sample_fn(self.model.model, (bsz, 1, len(self.genes)), **kwargs)
        return sample.squeeze(1).detach().cpu().numpy().astype(np.float32)
