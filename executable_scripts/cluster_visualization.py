import os
from datetime import datetime
from omegaconf import OmegaConf
import pickle

import numpy as np
import matplotlib.pyplot as plt

from source.experiments.eval_utils import load_concept_activation
from source.experiments.visualize_clusters import show_train_patches
from source.data.imagenet import create_dataset


def main(cfg):

    vis_selection_file = os.path.join(
        cfg.visualization_dir, f"{cfg.run_id_align}_{cfg.run_id}.pkl"
    )
    with open(vis_selection_file, "rb") as f:
        vis_selection_dict = pickle.load(f)

    config_path = [
        file_path[0]
        for file_path in os.walk(
            os.path.join(cfg.exp_dir, str(cfg.run_id), "job_results")
        )
        if "config.yaml" in file_path[2]
    ][0]
    result_path = [
        file_path[0]
        for file_path in os.walk(
            os.path.join(
                cfg.exp_dir, str(cfg.run_id), "job_results", "clustering", "results"
            )
        )
        if "sample_idx.npy" in file_path[2]
    ][0]
    print("run config path", config_path)
    print("run result path", result_path)

    if cfg.cls:
        token_idx = None
    else:
        token_idx = load_concept_activation(
            config_path,
            None,
            train=False,
            cluster_assignment="",
            filename_root="token_idx.npy",
            take_parent=False,
        )
    soft_assignments = load_concept_activation(
        config_path,
        None,
        train=True,
        cluster_assignment="hdbscan",
        filename_root="clustering.npy",
        take_parent=False,
    )
    hard_assignments = load_concept_activation(
        config_path,
        None,
        train=True,
        cluster_assignment="hard_clustering",
        filename_root="clustering.npy",
        take_parent=False,
    )

    print(soft_assignments.shape, hard_assignments.shape)

    # get samples idx from dataset creation
    config_file = os.path.join(config_path, "config.yaml")
    cfg_data = OmegaConf.load(config_file).dataset
    cfg_data.params.root = cfg.data_root
    cfg_data.params.feature_layer = 0
    dataset, _ = create_dataset(
        cfg_data, return_label=True, cuda=False, train=True, indices_subsample=None
    )
    sample_idx = dataset.indices
    # repeat as often as token were selected from one image
    if not cfg.cls and int(cfg_data.subsample_ratio * 121) > 1:
        sample_idx = np.repeat(
            sample_idx, repeats=int(cfg_data.subsample_ratio * 196 / 49)
        )
    print(sample_idx)

    n_cols = cfg.n_examples // cfg.n_rows

    visualization_dir = os.path.join(
        cfg.visualization_dir, str(cfg.run_id_align), str(cfg.run_id)
    )
    os.makedirs(visualization_dir)

    for meta_cluster_idx in vis_selection_dict:
        vis_selection = vis_selection_dict[meta_cluster_idx]
        for i, cluster_idx in enumerate(vis_selection):
            if cluster_idx is None:
                continue
            fig, ax = plt.subplots(
                cfg.n_rows, n_cols, figsize=(5 * n_cols // 2, 5 * n_cols // 2)
            )
            ax = ax.flatten()
            show_train_patches(
                cluster_idx,
                soft_assignments,
                hard_assignments,
                token_idx=token_idx,
                sample_idx=sample_idx,
                n_samples=cfg.n_examples,
                cfg_data=cfg_data,
                n_patches=121,
                random=False,
                title=False,
                ax=ax,
            )
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    visualization_dir,
                    f"{cfg.run_id_align}_{cfg.run_id}_{meta_cluster_idx}_{i}_{cluster_idx}.svg",
                )
            )
            plt.close()


if __name__ == "__main__":
    base_conf = OmegaConf.load("./source/conf/cluster_visualization.yaml")
    cli_conf = OmegaConf.from_cli()
    now = datetime.now()
    now_conf = OmegaConf.create({"now_dir": f"{now:%Y-%m-%d}/{now:%H-%M-%S}"})
    # merge them all
    conf = OmegaConf.merge(now_conf, base_conf, cli_conf)
    main(conf)
