import ast
import os
from pathlib import Path
from copy import copy
from omegaconf import OmegaConf

import pandas as pd
from pandas import json_normalize
import numpy as np
import pickle

from source.data.imagenet import create_dataset


### Loading results ###
def unnest_dataframe(df, column, mode="concat"):
    df_normalized = json_normalize(df[column])
    df_normalized.index = df.index
    if mode == "concat":
        # df_unnested = pd.concat([df.loc[:,~df.columns.isin(df_normalized.columns)], df_normalized], axis=1)#.drop(column, axis=1)
        df_unnested = pd.concat([df, df_normalized], axis=1)  # .drop(column, axis=1
    elif mode == "combine":
        df_unnested = df.copy()
        for col in df_normalized:
            if col in df_unnested:
                df_unnested[col] = df_normalized[col].combine_first(df_unnested[col])
            else:
                df_unnested[col] = np.nan

    return df_unnested  # ,df_normalized


def run_id_from_ckpt_path(ckpt_path, only_id=True):
    if type(ckpt_path) is not str:
        return
    if "/" in ckpt_path:
        model_id = ckpt_path.split("/")[-2].split(":")[0]  # model-id
        if only_id:
            return model_id.split("-")[1]
        else:
            return model_id
    else:
        return ckpt_path


def load_configs_df(exp_dir, start_date, measured: list):
    """
    If configs are not stored in result df.
    """
    config_paths = []
    configs = []

    # Walk through nested directories to find 'config.yaml' files
    for dirpath, dirnames, filenames in os.walk(exp_dir):
        if "config.yaml" in filenames:
            config_path = os.path.join(dirpath, "config.yaml")
            # dirpath = os.path.relpath(dirpath, exp_dir)
            config_paths.append(dirpath)
            config = OmegaConf.load(config_path)
            configs.append(config)

    # Create a DataFrame with paths and configurations
    df = pd.DataFrame({"config_path": config_paths, "config": configs})

    # unnest config
    df["exp_config"] = df["config"].apply(lambda cfg: OmegaConf.to_container(cfg))
    df = unnest_dataframe(df, "exp_config")

    # time filter
    def get_run_date_time(config_path):
        config_path = config_path.split("/")
        return config_path[-2] + "_" + config_path[-1]

    df["run_date"] = pd.to_datetime(
        df["config_path"].apply(get_run_date_time), format="%Y-%m-%d_%H-%M-%S"
    )
    time_mask = df["run_date"] >= pd.to_datetime(start_date, format="%Y-%m-%d_%H-%M-%S")
    df = df[time_mask]

    print("config df shape", df.shape)

    # some re-naming
    df["vcl"] = df.apply(
        lambda row: f"{row['vcl.name']}_{row['vcl.cluster.discovery']}_{row['vcl.cluster.min_cluster_size']}-{row['vcl.cluster.min_samples']}_{row['dataset.subset_index']}-{row['dataset.subsample_ratio']}-{row['dataset.subsample_sample_ratio']}",
        axis=1,
    )
    df["representation_model"] = df["dataset.params.representation_model_ckpt"]
    df["feature_layer"] = df["dataset.params.feature_layer"]

    # load measures from run_info.yaml file
    if len(measured) > 0:
        for m in measured:
            df[m] = np.nan
        for idx in df.index:
            config_path = df.loc[idx, "config_path"]
            try:
                run_info = OmegaConf.to_container(
                    OmegaConf.load(os.path.join(config_path, "run_info.yaml"))
                )
                for m in measured:
                    if m in run_info:
                        df.loc[idx, m] = run_info[m]
            except:
                pass
    return df


def load_discovery_result(config_path, dummy, name="validity_train.pkl"):
    parts = config_path.split(os.sep)
    config_dir = os.sep.join(parts[:-2])
    exp_run_dir = os.sep.join(parts[-2:])
    filename = name
    file = Path(config_dir) / "results" / exp_run_dir / filename
    try:
        with open(str(file), "rb") as f:
            result = pickle.load(f)
    except:
        result = None
    return result


def load_concept_activation(
    config_path,
    dummy,
    cluster_assignment="",
    filename_root="clustering_batch.npy",
    train=False,
    take_parent=True,
):
    parts = config_path.split(os.sep)
    config_dir = os.sep.join(parts[:-2])
    exp_run_dir = os.sep.join(parts[-2:])
    pre = "train_" if train else ""
    filename = (
        f"{filename_root}"
        if cluster_assignment == ""
        else f"{pre}{cluster_assignment}-{filename_root}"
    )
    file = Path(config_dir) / "results" / exp_run_dir / filename
    activations = np.load(file, allow_pickle=True)
    return activations


def load_concept_activations(
    df_res,
    config_dir,
    cluster_assignment="",
    filename_root="clustering_batch.npy",
    train=False,
    take_parent=True,
):
    ca_dict_all = {}
    for idx in df_res.index:
        idx = idx if type(idx) is tuple else (idx,)
        ca_dict_all[(cluster_assignment, *idx)] = load_concept_activation(
            df_res["config_path"].loc[idx],
            config_dir,
            cluster_assignment,
            filename_root,
            train,
            take_parent,
        )

    return ca_dict_all


def load_alignment_results(exp_dir, start_date="2024-04-22_16-00-00"):
    def check_header(file_path, max_lines=3):
        with open(file_path, "r") as f:
            lines = [next(f).strip() for _ in range(max_lines)]
        # return lines
        if "groundtruth" in lines[0]:
            return 1
        else:
            return 3

    # load config files
    align_configs = {}
    align_results = {}
    empty_exp = []
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file == "config.yaml":
                align_configs[root] = OmegaConf.to_container(
                    OmegaConf.load(os.path.join(root, file))
                )
            elif file == "alignment.csv":
                file_path = os.path.join(root, file)
                header_lines = check_header(file_path)
                align_df = pd.read_csv(
                    file_path, index_col=[0, 1, 2], header=list(range(header_lines))
                )
                # reset index and columns to keep only featre layers
                if align_df.shape[0] > 0:
                    if align_df.isna().all().all():  # exclude all nans
                        empty_exp.append(root)
                    else:
                        align_df = align_df.reset_index(level=[0, 1], drop=True)
                        if header_lines == 3:
                            align_df = align_df.T.reset_index(
                                level=[0, 1], drop=True
                            ).T  # drop model and vcl name
                        align_df.index = align_df.index.astype(float)
                        try:
                            align_df.columns = (
                                align_df.columns.astype(float)
                                if not "groundtruth" in align_df.columns
                                else align_df.columns
                            )
                        except:
                            pass
                        align_results[root] = align_df
                else:
                    empty_exp.append(root)
            elif file == "alignment_cw.pkl":
                with open(os.path.join(root, file), "rb") as f:
                    align_results[root] = pickle.load(f)

    # delete empty runs from config df
    for root in empty_exp:
        del align_configs[root]

    align_configs = pd.DataFrame.from_dict(align_configs).T
    align_configs.index.name = "run_dir"
    align_configs = align_configs.reset_index().set_index(
        [
            "vcl",
            "vcl2",
            "cluster_assignment1",
            "cluster_assignment2",
            "model1",
            "model2",
        ]
    )
    align_configs.rename({"": "hdb"}, axis=0, level=2, inplace=True)

    # time filtering
    align_configs["start_date"] = pd.to_datetime(
        align_configs["start_date"], format="%Y-%m-%d_%H-%M-%S"
    )
    time_mask = align_configs["start_date"] >= pd.to_datetime(
        start_date, format="%Y-%m-%d_%H-%M-%S"
    )
    align_configs = align_configs[time_mask]

    return align_configs, align_results


def get_cluster_assignment(ca_dict_all):
    cluster_assignment_all = {}

    for k in ca_dict_all:
        print(k)
        cluster_assignment_df = {}
        ca_dict = ca_dict_all[k]

        for model in ca_dict:
            print(model)
            cluster_assignment_df_m = {}

            for fl in ca_dict[model]:
                cluster_probabilities = ca_dict[model][fl]
                cluster_labels = cluster_probabilities.argmax(axis=1)
                cluster_assignment_df_m[fl] = cluster_labels
            cluster_assignment_df[model] = pd.DataFrame(cluster_assignment_df_m)

        cluster_assignment_df = pd.concat(cluster_assignment_df, axis=1)
        cluster_assignment_all[k] = cluster_assignment_df

    return cluster_assignment_all


def process_validity(
    validity_dict,
    representation="embedded",
    what="overall",
    assignment_level=False,
    assignment=False,
):
    """
    representation: embedded or original
    what: overall: overall validity index, average over all clusters weighted by cluster size
        overall_cluster: un-weighted average over all clusters
    """
    # TODO adapt for test set
    if validity_dict is None or representation not in validity_dict:
        return np.nan

    def compute_validity(validity_dict_):
        if what == "overall":
            return validity_dict_[0]
        elif what == "overall_cluster":
            return np.average(validity_dict_[1])

    if assignment_level:
        if validity_dict is None:  # or assignment not in validity_dict:
            return np.nan
        else:
            result = {
                k: compute_validity(validity_dict[representation][k])
                for k in validity_dict[representation]
            }
            if assignment:
                result = result[assignment]
            return result
    else:
        return compute_validity(validity_dict[representation])


def load_groundtruth(
    df,
    exp_dir,
    gt_type,
    n_patches=121,
    data_root="/mnt/ImageNet-complete",
    meta_class_file="./data/imagenet_meta_classes.csv",
):
    if gt_type == "gt" or gt_type == "gt_meta":
        cfg_data = df.iloc[0]["config"].dataset
        cfg_data.params.root = data_root
        dataset, _ = create_dataset(
            cfg_data, return_label=True, cuda=False, train=True, indices_subsample=None
        )
        sample_idx = dataset.indices
        # repeat as often as token were selected from one image
        if (
            not cfg_data.params.remove_sequence
            and int(cfg_data.subsample_ratio * n_patches) > 1
        ):  # TODO compute n_token from hyperparam
            sample_idx = np.repeat(
                sample_idx, repeats=int(cfg_data.subsample_ratio * n_patches)
            )
        gt = np.array(dataset.dataset.targets)[sample_idx]
        if gt_type == "gt_meta":
            # load imagenet meta classes
            in_meta_classes = pd.read_csv(meta_class_file, index_col=0)
            meta_label_mapping = (
                in_meta_classes.set_index("class_num")["meta_class_num"]
                .astype(int)
                .to_dict()
            )
            unknown_label = max(meta_label_mapping.values()) + 1
            gt_meta_labels = []
            for label in gt:
                meta_label = (
                    meta_label_mapping[label]
                    if label in meta_label_mapping
                    else unknown_label
                )
                gt_meta_labels.append(meta_label)
            gt = np.array(gt_meta_labels)
    elif gt_type == "loc":
        gt = load_concept_activations(
            df,
            exp_dir,
            train=False,
            cluster_assignment="",
            filename_root="token_idx.npy",
            take_parent=False,
        )
        gt = list(gt.values())[0]

    unique_label = np.unique(gt)
    n_label = len(unique_label)
    n_samples = len(gt)
    class_label_idx_map = {k: i for i, k in enumerate(unique_label)}
    class_label_idx = [class_label_idx_map[k] for k in gt]
    class_label_assignment = np.zeros((n_samples, n_label), dtype=int)
    for i in range(n_samples):
        class_label_assignment[i, class_label_idx[i]] = 1.0
    return class_label_assignment
