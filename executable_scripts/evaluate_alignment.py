import logging
import os
from omegaconf import OmegaConf
from datetime import datetime
import wandb
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

from source.experiments.eval_utils import (
    load_configs_df,
    load_concept_activations,
    load_groundtruth,
)
from source.experiments.alignment import (
    hullermeier_fuzzy_rand,
    l1_difference_torch_clusterwise,
)


def concept_alignment_clusterwise(cfg):
    # load configs for all runs
    measured = [
        "n_cluster",
    ]
    df = load_configs_df(
        exp_dir=cfg.cfg_dir, start_date=cfg.start_date, measured=measured
    )
    df.set_index(["vcl", "representation_model", "feature_layer"], inplace=True)
    # drop dupliated
    df = df.loc[~df.index.duplicated()]

    df1 = df.loc[
        pd.IndexSlice[
            [
                cfg.vcl,
            ],
            [
                cfg.model1,
            ],
            [cfg.fl1, cfg.fl2],
        ]
    ]
    concept_activation_dict_1 = load_concept_activations(
        df1,
        cfg.cfg_dir,
        cluster_assignment=cfg.cluster_assignment1,
        filename_root=cfg.filename_root,
        train=cfg.train_set,
        take_parent=cfg.take_parent,
    )
    df2 = df.loc[
        pd.IndexSlice[
            [
                cfg.vcl2,
            ],
            [
                cfg.model2,
            ],
            [cfg.fl2],
        ]
    ]
    concept_activation_dict_2 = load_concept_activations(
        df2,
        cfg.cfg_dir,
        cluster_assignment=cfg.cluster_assignment2,
        filename_root=cfg.filename_root,
        train=cfg.train_set,
        take_parent=cfg.take_parent,
    )

    # subsample samples
    jump_rate = int(1 / cfg.subsample_token_ratio)
    concept_activation_1 = concept_activation_dict_1[
        (cfg.cluster_assignment1, cfg.vcl, cfg.model1, cfg.fl1)
    ][::jump_rate]
    concept_activation_2 = concept_activation_dict_2[
        (cfg.cluster_assignment2, cfg.vcl2, cfg.model2, cfg.fl2)
    ][::jump_rate]

    alignment, cluster_match = l1_difference_torch_clusterwise(
        concept_activation_1, concept_activation_2, device="cuda" if cfg.cuda else "cpu"
    )
    result_dict = {"alignment": alignment, "cluster_match": cluster_match}

    # save
    out_dir = os.path.join(cfg.exp_dir, "concept_alignment", cfg.now_dir)
    os.makedirs(out_dir)
    cfg_file = os.path.join(out_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_file)
    result_file = os.path.join(out_dir, "alignment_cw.pkl")
    with open(result_file, "wb") as f:
        pickle.dump(result_dict, f)
    # save
    df["config_path"].to_csv(os.path.join(out_dir, "run_configs.csv"))


def concept_alignment(cfg):
    # compute concept alignment between layers of two models

    # load concept activations
    # load configs for all runs
    measured = [
        "n_cluster",
    ]
    df = load_configs_df(
        exp_dir=cfg.cfg_dir, start_date=cfg.start_date, measured=measured
    )

    indexers = ["vcl", "representation_model", "feature_layer"]
    df.set_index(indexers, inplace=True)
    if not cfg.train_set:
        print("filtering for n_classes test")
        mask = df["dataset_test.params.n_classes"] == cfg.n_classes_test
        df = df.loc[mask]
    # drop dupliated
    # keep the latest run
    df.sort_values("now_dir", ascending=False, inplace=True)
    if cfg.numerate_duplicates:
        df["count"] = df.groupby(level=[0, 1, 2]).cumcount()
        df.reset_index(inplace=True)
        df["vcl"] = df[["vcl", "count"]].apply(
            lambda row: row["vcl"] + "_" + str(row["count"]), axis=1
        )
        df = df.set_index(indexers)
    else:
        df = df.loc[~df.index.duplicated(keep="first")]

    # extract relevant configs
    df1 = df.loc[
        pd.IndexSlice[
            [
                cfg.vcl,
            ],
            [
                cfg.model1,
            ],
            :,
        ]
    ]
    print(df1)
    concept_activation_dict_1 = load_concept_activations(
        df1,
        cfg.cfg_dir,
        cluster_assignment=cfg.cluster_assignment1,
        filename_root=cfg.filename_root,
        train=cfg.train_set,
        take_parent=cfg.take_parent,
    )
    concept_activation_dict_1 = {
        k: concept_activation_dict_1[k]
        for k in concept_activation_dict_1
        if concept_activation_dict_1[k] is not None
    }
    print(cfg.model2)
    if cfg.model2 == "groundtruth":
        print("GROUNDTRUTH")
        df2 = df1
        n_samples = concept_activation_dict_1[
            list(concept_activation_dict_1.keys())[0]
        ].shape[0]
        concept_activation_dict_2 = {
            "groundtruth": load_groundtruth(
                df1.iloc[[0]], exp_dir=cfg.cfg_dir, gt_type=cfg.groundtruth_type
            )
        }
    else:
        df2 = df.loc[
            pd.IndexSlice[
                [
                    cfg.vcl2,
                ],
                [
                    cfg.model2,
                ],
                :,
            ]
        ]
        print(df2)
        concept_activation_dict_2 = load_concept_activations(
            df2,
            cfg.cfg_dir,
            cluster_assignment=cfg.cluster_assignment2,
            filename_root=cfg.filename_root,
            train=cfg.train_set,
            take_parent=cfg.take_parent,
        )
        concept_activation_dict_2 = {
            k: concept_activation_dict_2[k]
            for k in concept_activation_dict_2
            if concept_activation_dict_2[k] is not None
        }

    print(
        [
            concept_activation_dict_1[k].shape
            for k in concept_activation_dict_1
            if len(concept_activation_dict_1[k].shape) > 0
        ]
    )
    print(
        [
            concept_activation_dict_2[k].shape
            for k in concept_activation_dict_2
            if len(concept_activation_dict_2[k].shape) > 0
        ]
    )

    # subsample samples
    jump_rate = int(1 / cfg.subsample_token_ratio)
    concept_activation_dict_1 = {
        k: concept_activation_dict_1[k][::jump_rate] for k in concept_activation_dict_1
    }
    concept_activation_dict_2 = {
        k: concept_activation_dict_2[k][::jump_rate] for k in concept_activation_dict_2
    }

    if cfg.use_l1:

        def norm_activation(activation):
            return activation / activation.sum(axis=1, keepdims=True)

        if cfg.cluster_assignment1 == "projection":
            print("norming concept activation 1")
            concept_activation_dict_1 = {
                k: norm_activation(
                    np.clip(concept_activation_dict_1[k], a_min=0.0, a_max=1.0)
                )
                for k in concept_activation_dict_1
            }
        if cfg.cluster_assignment2 == "projection":
            print("norming concept activation 2")
            concept_activation_dict_2 = {
                k: norm_activation(
                    np.clip(concept_activation_dict_2[k], a_min=0, a_max=1.0)
                )
                for k in concept_activation_dict_2
            }

    # compute concept similarity across layer
    df_sim = pd.DataFrame(
        index=df1.index,
        columns=["groundtruth", "loc"] if cfg.model2 == "groundtruth" else df2.index,
    )
    for fl1 in tqdm(df_sim.index):
        for fl2 in df_sim.columns:
            if fl1[-1] != fl2[-1] and cfg.only_same_layer:
                continue
            if (
                not cfg.model2 == "groundtruth"
                and int(fl1[-1]) != (int(fl2[-1]) - 1)
                and cfg.only_next_layer
            ):
                continue

            fl1_key = (cfg.cluster_assignment1, *fl1)  # (cfg.model1,fl1)
            fl2_key = (
                (cfg.cluster_assignment2, *fl2)
                if (fl2 not in ["groundtruth", "loc"])
                else fl2
            )

            if (
                fl1_key in concept_activation_dict_1
                and fl2_key in concept_activation_dict_2
            ):
                hfr = hullermeier_fuzzy_rand(
                    concept_activation_dict_1[fl1_key],
                    concept_activation_dict_2[fl2_key],
                    l1_dist=cfg.use_l1,
                    crisp=cfg.crisp,
                )
                df_sim.loc[fl1, fl2] = hfr

    df_sim = df_sim.sort_index(axis=0).sort_index(axis=1)

    # save
    out_dir = os.path.join(cfg.exp_dir, "concept_alignment", cfg.now_dir)
    os.makedirs(out_dir)
    cfg_file = os.path.join(out_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_file)
    result_file = os.path.join(out_dir, "alignment.csv")
    df_sim.to_csv(result_file)
    # save
    df["config_path"].to_csv(os.path.join(out_dir, "run_configs.csv"))


def main_evaluate_concept_alignment(cfg):
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    logging.basicConfig(level=logging.INFO)
    wandb.init(
        project=cfg.wandb_project_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=True,
    )
    wandb.log({"run_id": int(cfg.run_id)})

    if cfg.clusterwise:
        concept_alignment_clusterwise(cfg)
    else:
        concept_alignment(cfg)


if __name__ == "__main__":
    base_conf = OmegaConf.load("./source/conf/evaluate_alignment.yaml")
    cli_conf = OmegaConf.from_cli()
    now = datetime.now()
    now_conf = OmegaConf.create({"now_dir": f"{now:%Y-%m-%d}/{now:%H-%M-%S}"})
    # merge them all
    conf = OmegaConf.merge(now_conf, base_conf, cli_conf)
    main_evaluate_concept_alignment(conf)
