import logging
import os
import pickle
from datetime import datetime
from omegaconf import OmegaConf
import wandb

import multiprocessing
import numpy as np
import pandas as pd
from hdbscan.validity import validity_index

from source.data.utils import preload_dataset
from source.experiments.utils import load_data


from mcd.virtual_concept_layer import StaticVCL
from source.experiments.dr_evaluation import (
    RMSE_batchwise,
    intrinsic_dimensionality,
    measure_linearity,
)


def merge_configs(train_config, test_config):
    def recursive_merge(train, test):
        for key, value in train.items():
            if isinstance(value, dict) and key in test:
                recursive_merge(value, test[key])
            elif key not in test or (test[key] is None or test[key] == "None"):
                test[key] = value

    recursive_merge(OmegaConf.to_container(train_config, resolve=True), test_config)
    return test_config


def main_fit_clustering(cfg) -> None:
    logger = logging.getLogger(__name__)

    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    now_dir = cfg.now_dir
    exp_dir = cfg.exp_dir
    cfg_dir = f"{exp_dir}/clustering/{now_dir}"
    result_dir = f"{exp_dir}/clustering/results/{now_dir}"
    os.makedirs(cfg_dir)
    os.makedirs(result_dir)

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    wandb.init(
        project=cfg.wandb_project_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=True,
    )

    # Log the configuration
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    wandb.log({"run_id": int(cfg.run_id)})

    # search for previous run with hdbscan to extract number of clusters from there
    if cfg.vcl.cluster.discovery != "hdbscan" and cfg.vcl.cluster.n_cluster < 0:
        # load n_concepts file
        df_ncluster = pd.read_csv(cfg.experiment.ncluster_file, index_col=[0, 1, 2])

        look_up_discovery = f"umap_hdbscan_50-20_1-1.0-0.25"
        cfg.vcl.cluster.n_cluster = df_ncluster.loc[
            look_up_discovery,
            cfg.dataset.params.representation_model_ckpt,
            cfg.dataset.params.feature_layer,
        ].item()
        logger.info(
            f"Retrieved number of concepts {cfg.vcl.cluster.n_cluster} from previous run."
        )
        if cfg.vcl.cluster.discovery == "pca":
            cfg.vcl.cluster.n_cluster = 768

    # preload dataset
    logger.info("Loading train data.")
    np.random.seed(42)
    loader = load_data(
        cfg.dataset,
        batch_size=cfg.measures.batch_size,
        return_label=False,  # return idx instead of label
        shuffle=False,
        train=True,
        numpy=False,
    )
    n_samples = len(loader.dataset)
    loader = preload_dataset(loader, batch_size=None)
    batch = next(iter(loader))
    if len(batch) == 3:
        X, token_idx, sample_idx = batch
    else:
        X, sample_idx = batch
        token_idx = None
    del batch

    # fit clustering
    logger.info(f"Fitting clustering with {X.shape[0]} samples.")
    vcl = StaticVCL(cfg.vcl)
    vcl.fit(X)

    # save config and results
    cfg_file = f"{cfg_dir}/config.yaml"
    OmegaConf.save(cfg, cfg_file)
    if cfg.experiment.save_clustering:
        vcl.save_clustering(os.path.join(result_dir, "clustering.pkl"))
        logger.info(f"Saved clustering.")

    if cfg.experiment.ndlr_metrics and cfg.vcl.name != "ident":
        rmse = RMSE_batchwise(X, vcl.embedded_x)
        wandb.log({"rmse": float(rmse)})
    else:
        rmse = np.nan

    # dump info on runs
    run_info = {
        "n_cluster": vcl.vcl_cfg.cluster.n_cluster,
        "n_sample": X.shape[0],
        "reconstruction_loss": float(vcl.reconstruction_loss),
        "rmse": float(rmse),
    }

    if cfg.vcl.cluster.discovery == "hdbscan":
        n_noise = int(np.logical_not(vcl.noise_mask).sum())
        cluster_members = int(X.shape[0] - n_noise)
        logger.info(
            f"HDBSCAN resulted in: {vcl.vcl_cfg.cluster.n_cluster} with {cluster_members} cluster members and {n_noise} noisy samples."
        )
        run_info.update({"n_noise": n_noise})
        wandb.log({"noise_ratio": n_noise / X.shape[0]})
        wandb.log({"reconstruction_loss": float(vcl.reconstruction_loss)})
    wandb.log({"n_cluster": vcl.vcl_cfg.cluster.n_cluster})

    # save
    run_info = OmegaConf.create(run_info)
    info_file = f"{cfg_dir}/run_info.yaml"
    OmegaConf.save(run_info, info_file)
    logger.info(f"Saved results.")

    if token_idx is not None:
        np.save(os.path.join(result_dir, f"token_idx.npy"), token_idx.cpu().numpy())
        logger.info(f"Saved token idx.")

    vcl.cluster_assignment(X)
    vcl.save_assignment(result_dir, train=True)
    np.save(os.path.join(result_dir, f"sample_idx.npy"), sample_idx.cpu().numpy())
    logger.info(f"Saved sample idx.")

    # measure linearity
    if cfg.experiment.eval_linearity:
        lin_acc, lin_auc = measure_linearity(X.numpy(), vcl.labels)
        lin_acc_soft, lin_auc_soft = measure_linearity(
            X.numpy(), vcl.ca[list(vcl.ca.keys())[0]].argmax(axis=1)
        )
        wandb.log({"linearity": float(lin_acc)})
        wandb.log({"linearity_soft": float(lin_acc_soft)})
        wandb.log({"linearity_auc": float(lin_auc)})
        wandb.log({"linearity_auc_soft": float(lin_auc_soft)})
        with open(os.path.join(result_dir, "linearity.pkl"), "wb") as f:
            pickle.dump(
                {
                    "linearity": float(lin_acc),
                    "linearity_soft": float(lin_acc_soft),
                    "linearity_auc": float(lin_auc),
                    "linearity_auc_soft": float(lin_auc_soft),
                },
                f,
            )

    # compute validity score
    if cfg.experiment.compute_train_validity and vcl.vcl_cfg.cluster.n_cluster > 10:
        logger.info("Computing validty.")
        validity_score = {}
        embedded_x = vcl.embedded_x
        validity_score["embedded"] = validity_index(
            embedded_x.astype(np.float64),
            vcl.labels,
            per_cluster_scores=True,
            metric=cfg.vcl.cluster.metric,
        )

        wandb.log({"validity_embedded": float(validity_score["embedded"][0])})
        wandb.log(
            {
                "validity_cluster_embedded": float(
                    np.average(validity_score["embedded"][1])
                )
            }
        )

        # save
        with open(os.path.join(result_dir, "validity_train.pkl"), "wb") as f:
            pickle.dump(validity_score, f)

    if cfg.experiment.compute_train_id:
        # compute intrinsic dimensionality:
        logger.info("Computing concept-wise ID.")
        intrinsic_dimensionality_scores = {}
        for k in [
            "hard_clustering",
        ]:
            logger.info(f"for {k}")
            label = vcl.ca[k]
            if label.ndim == 2:
                label = label.argmax(axis=1)
            intrinsic_dimensionality_scores[k] = intrinsic_dimensionality(
                vcl.embedded_x,
                label,
                clusterwise=True,
                method="twonn",
                discard_fraction=0.5,
                data_discard_factor=cfg.measures.id_estimation_data_dicard_factor,
            )
        logger.info("Computing global ID.")
        intrinsic_dimensionality_scores["union"] = intrinsic_dimensionality(
            vcl.embedded_x,
            None,
            clusterwise=False,
            method="twonn",
            discard_fraction=0.5,
            data_discard_factor=cfg.measures.id_estimation_data_dicard_factor * 10,
        )

        # save
        with open(os.path.join(result_dir, "intrinsic_dimension_train.pkl"), "wb") as f:
            pickle.dump(intrinsic_dimensionality_scores, f)


if __name__ == "__main__":
    base_conf = OmegaConf.load("./source/conf/concept_discovery.yaml")
    cli_conf = OmegaConf.from_cli()
    now = datetime.now()
    now_conf = OmegaConf.create({"now_dir": f"{now:%Y-%m-%d}/{now:%H-%M-%S}"})
    # merge them all
    conf = OmegaConf.merge(now_conf, base_conf, cli_conf)

    multiprocessing.set_start_method("spawn", force=True)

    main_fit_clustering(conf)
