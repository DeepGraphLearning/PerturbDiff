"""CLI for inference-only PerturbDiff prediction from a control h5ad file."""

import argparse
import logging
import os
import sys

exc_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(exc_dir)

from src.apps.inference import PerturbDiffPredictor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selected-gene-file", required=True)
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--output-h5ad", required=True)
    parser.add_argument("--perturbation", required=True)
    parser.add_argument("--cell-type", required=True)
    parser.add_argument("--batch-key", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--gene-column", default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--celltype-embedding-path", default=None)
    parser.add_argument("--gene-embedding-path", action="append", default=None)
    parser.add_argument("--pert-embedding-path", default=None)
    parser.add_argument("--drug-embedding-path", default=None)
    parser.add_argument("--replogle-gene-embedding-path", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-time", type=int, default=100)
    parser.add_argument("--normalize-counts", type=float, default=10.0)
    parser.add_argument("--min-gene-overlap", type=float, default=0.8)
    parser.add_argument("--ddpm", action="store_true", help="Use DDPM instead of DDIM.")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    covariate_asset_paths = {
        "celltype_embedding_path": args.celltype_embedding_path,
        "gene_embedding_path": args.gene_embedding_path,
        "pert_embedding_path": args.pert_embedding_path,
        "drug_embedding_path": args.drug_embedding_path,
        "replogle_gene_embedding_path": args.replogle_gene_embedding_path,
    }
    covariate_asset_paths = {k: v for k, v in covariate_asset_paths.items() if v is not None}
    predictor = PerturbDiffPredictor(
        checkpoint_path=args.checkpoint,
        selected_gene_file=args.selected_gene_file,
        device=args.device,
        covariate_asset_paths=covariate_asset_paths or None,
    )
    pred = predictor.predict_file(
        input_h5ad=args.input_h5ad,
        output_h5ad=args.output_h5ad,
        perturbation=args.perturbation,
        cell_type=args.cell_type,
        batch_key=args.batch_key,
        dataset_name=args.dataset_name,
        gene_column=args.gene_column,
        layer=args.layer,
        normalize_counts=args.normalize_counts,
        batch_size=args.batch_size,
        start_time=args.start_time,
        use_ddim=not args.ddpm,
        progress=args.progress,
        min_gene_overlap=args.min_gene_overlap,
    )
    print(f"Wrote {args.output_h5ad}: shape={pred.shape}, min={pred.X.min():.4f}, max={pred.X.max():.4f}")


if __name__ == "__main__":
    main()
