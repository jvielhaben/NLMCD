import networkx as nx
import numpy as np


from functools import partial

import numpy as np
import timm
import torch
import torch.nn
from torch.utils.data import DataLoader
import torchvision
from source.data.imagenet import (
    CHATGPT30_CLASSES,
    CHATGPT50_CLASSES,
    CIFAR10_CLASSES,
    RESNET_FEATUREMAP_WIDTH,
    SubsetIDX,
    ThroughModel_in,
    get_model_architecture_in,
    stratified_subsets,
)
from source.data.utils import (
    DimensionTransformer,
    PatchDivide,
    Squeeze,
    Unsqueeze,
    dimension_trafo_collate_fn,
    numpy_collate_fn,
)


def get_trafos_in(cfg, model, cuda=True):
    # dimension transformer
    # depends on architecture
    # pass through model
    if len(cfg.params.representation_model_ckpt) > 0 and cfg.params.feature_layer > 0:
        if cuda:
            model = model.cuda()
    # to avoid saving buffers
    for param in model.parameters():
        param.requires_grad = False

    # models's imagenet transformations
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    transforms = transforms.transforms

    if cfg.params.feature_layer > 0:
        model = ThroughModel_in(
            model,
            cfg.params.feature_layer,
            remove_cls=cfg.params.remove_cls,
            remove_sequence=cfg.params.remove_sequence,
            pool_token=cfg.params.pool_token,
            kernel_size=cfg.params.kernel_size,
            pre_layernorm=cfg.params.pre_layernorm,
            mlp=cfg.params.mlp,
        )
        transforms.append(Unsqueeze())
        transforms.append(model)
        transforms.append(Squeeze())
        architecture = get_model_architecture_in(str(model))
        if architecture == "resnet":
            # pool resnet featurelayer to ViT feature layer size of 14x14 for patch-comparability
            kernel_size = RESNET_FEATUREMAP_WIDTH[cfg.params.feature_layer] // 14
            if kernel_size > 1:
                transforms.append(
                    torch.nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
                )
        if cfg.params.flatten:
            if architecture == "vit":
                dimension_trafo_load, dimension_trafo_collate = "", "0,1,2->01,2"  # ViT
            elif architecture == "resnet":
                dimension_trafo_load, dimension_trafo_collate = (
                    "",
                    "0,1,2,3->023,1",
                )  # CNN
        else:
            dimension_trafo_load, dimension_trafo_collate = "", ""
    else:
        # transforms = [torchvision.transforms.Resize(248), torchvision.transforms.CenterCrop(size=(224,224)),torchvision.transforms.ToTensor()]
        if cfg.params.input_patch_size > 0:
            # transforms.extend([PatchDivide(patch_size=cfg.input_patch_size, batchwise=False),Squeeze()])
            transforms.append(
                PatchDivide(
                    patch_size=cfg.params.input_patch_size,
                    image_size=224,
                    n_channels=3,
                    batchwise=False,
                )
            )
            dimension_trafo_load, dimension_trafo_collate = "", "0,1,2,3,4->01,2,3,4"
        else:
            dimension_trafo_load, dimension_trafo_collate = "", ""

    if len(dimension_trafo_load) > 0:
        # TODO dimension trafo in datatset load not necessary for any experiments, remove?
        assert "->" in dimension_trafo_load
        transforms.append(DimensionTransformer(dimension_trafo_load))

    # potentially pooling

    return torchvision.transforms.Compose(transforms), dimension_trafo_collate


def create_dataset(
    cfg, model, dataset, return_label, cuda, train, indices_subsample=None
):
    # print('Creating dataset.')
    transforms, dimension_trafo_collate = get_trafos_in(cfg, model, cuda=cuda)
    # select subset of classes
    # class_idx = np.load(os.path.join(root, f'class_idx_{cfg.params.n_classes}.npy'))
    if cfg.params.n_classes == 1000:
        indices = np.arange(len(dataset), dtype=int)
    else:
        if cfg.params.n_classes == 10:
            print("CIFAR10 selection selection")
            class_idx = [dataset.class_to_idx[c] for c in CIFAR10_CLASSES]
        elif cfg.params.n_classes == 50:
            print("ChatGPT50 selection")
            class_idx = [dataset.class_to_idx[c] for c in CHATGPT50_CLASSES]
        elif cfg.params.n_classes == 30:
            print("ChatGPT50 selection")
            class_idx = [dataset.class_to_idx[c] for c in CHATGPT30_CLASSES]
        else:
            print("Class selection not implemented!")
        # print(len(class_idx))
        indices = np.where(np.isin(dataset.targets, class_idx))[0]

    if indices_subsample is None:
        # print('Class selection indeices', indices.shape)
        # np.random.seed(42)
        # indices_subsample  = np.random.choice(len(indices), size=int(cfg.subsample_sample_ratio*len(indices)), replace=False)

        indices_subsample = stratified_subsets(
            indices,
            np.array(dataset.targets)[indices],
            subset_index=cfg.subset_index,
            subsample_ratio=cfg.subsample_sample_ratio,
        )

    # print('Indices subsample shape', indices_subsample.shape)
    subset_dataset = SubsetIDX(
        dataset, indices=indices[indices_subsample], return_label=return_label
    )

    return subset_dataset, dimension_trafo_collate


def imagenet_loader(
    cfg,
    model,
    dataset,
    batch_size,
    train,
    return_label=True,
    numpy=False,
    shuffle=None,
    cuda=True,
    indices_subsample=None,
):
    dataset, dimension_trafo_collate = create_dataset(
        cfg,
        model,
        dataset,
        return_label,
        cuda,
        train,
        indices_subsample=indices_subsample,
    )

    # print('Creating dataloader.')
    if len(dimension_trafo_collate) > 0:
        dt_collate = DimensionTransformer(dimension_trafo_collate)
        dt_collate_idx = DimensionTransformer(dimension_trafo_collate)
        collate_fn = partial(
            dimension_trafo_collate_fn,
            dimension_transformer=dt_collate,
            dimension_transformer_idx=dt_collate_idx,
            numpy=numpy,
            subsample_ratio=cfg.subsample_ratio,
            center_sampling=cfg.center_sampling,
        )
    else:
        if numpy:
            collate_fn = numpy_collate_fn
        else:
            collate_fn = None

    if shuffle is None:
        shuffle = train

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=1,
        shuffle=shuffle,
    )

    return dataloader


def compute_raw_transition_scores(assignment1, assignment2):
    """
    Compute transitions between non noise clusters
    """
    # Get the number of clusters in each assignment
    # TODO use actual number including empty clusters
    num_clusters_1 = (
        np.concatenate(assignment1).max() + 1
    )  # len(np.unique(assignment1))
    num_clusters_2 = (
        np.concatenate(assignment2).max() + 1
    )  # len(np.unique(assignment2))

    # Create the transition matrix with zeros
    # print(num_clusters_1, num_clusters_2)
    transition_matrix = np.zeros((num_clusters_1, num_clusters_2), dtype=int)

    # print(transition_matrix.shape, assignment1.max(), assignment2.max())
    # Fill the transition matrix with the number of data points transitioning between clusters
    for i in range(len(assignment1)):
        clusters_1 = np.array(assignment1[i])
        clusters_2 = np.array(assignment2[i])
        np.add.at(transition_matrix, (clusters_1[:, None], clusters_2), 1)

    return transition_matrix


# %%
def threshold_assignment(clustering, threshold=0.1, hard=False):
    """
    Threshold a clustering assignment to obtain hard or soft assignments.

    Parameters
    ----------
    clustering: np.ndarray
        The clustering assignment. The first dimension should correspond to the samples and the second dimension should correspond to the clusters.
    threshold: float
        The threshold to use for thresholding the clustering probabilities.
    hard: bool
        Whether to return hard or soft assignments.

    Returns
    -------
    assignments: list(np.ndarray)
        The thresholded assignments. If hard is True, the first column will contain the indices and the second column will contain the hard assignments. If hard is False, the first column will contain the indices and the second column will contain the soft assignments

    """
    # Check if the input clustering is 2D
    if clustering.ndim != 2:
        raise ValueError("Input clustering must be 2D.")

    if hard:
        clustering_hard = clustering.argmax(axis=1)
        clustering_max = clustering[np.arange(clustering.shape[0]), clustering_hard]
        noise_mask = np.less_equal(clustering_max, threshold)
        clustering_hard[noise_mask] = -1
        # add indices range as the first column
        clustering_hard = clustering_hard[:, np.newaxis]
        return clustering_hard

    # Create a mask where clustering probabilities are above the threshold
    above_threshold_mask = clustering > threshold

    # Initialize an empty list to collect soft-assigned samples
    soft_assigned_samples = []

    # Iterate through each sample
    for i in range(clustering.shape[0]):
        # Get indices of all clusters above the threshold for the current sample
        clusters_above_threshold = np.nonzero(above_threshold_mask[i])[0]

        # Append (i, cluster) pairs for clusters above the threshold
        if clusters_above_threshold.size > 0:
            soft_assigned_samples.append(clusters_above_threshold)

        elif clusters_above_threshold.size > 1:
            print(clusters_above_threshold)
        else:
            soft_assigned_samples.append(
                [-1]
            )  # Assign noise (-1) if no clusters are above the threshold

    # Convert list to a NumPy array
    return soft_assigned_samples


def single_limit_transitions_to_graph(transitions, threshold):
    """
    Convert raw transitions between layers into a graph structure, keeping only the transitions where
    the contribution to the target node is at least threshold% of its total incoming transitions.

    Parameters:
    transitions: pd.Series
        A Pandas Series with a MultiIndex where each index is a tuple (layer1, layer2) indicating transitions between layers.
        Each element of the Series is a NumPy array of size (concepts_layer1, concepts_layer2) representing the transitions between clusters.
    threshold: float
        A threshold percentage (0-100) indicating the minimum contribution required for a transition to be included in the graph.

    Returns:
    graph: dict
        A dictionary where keys are tuples of the form (layer1, cluster_a_layer1) and values are lists of tuples
        of the form (layer2, cluster_b_layer2, transition_score) indicating the clusters in the next layer with transitions.
    """

    G = nx.DiGraph()

    for layer1, layer2 in reversed(transitions.index):
        transition_matrix = transitions.loc[layer1, layer2]
        # transition_matrix = transition_matrix[:-1, :-1]

        # Calculate the total incoming transitions for each cluster in layer2
        total_incoming_transitions = transition_matrix.sum(axis=0)

        for cluster1 in range(transition_matrix.shape[0] - 1):
            incoming_edges = []

            for cluster2 in range(transition_matrix.shape[1] - 1):
                G.add_node((layer2, cluster2))
                transition_score = transition_matrix[cluster1, cluster2]
                total_incoming = total_incoming_transitions[cluster2]

                # Calculate the contribution percentage for this transition
                if total_incoming > 0:
                    contribution_percentage = transition_score / total_incoming

                    if contribution_percentage >= threshold:
                        incoming_edges.append(
                            ((layer1, cluster1), (layer2, cluster2), transition_score)
                        )
            # Sort the incoming edges based on transition scores in descending order
            incoming_edges.sort(key=lambda x: x[2], reverse=True)
            # Add edges to the graph
            for source, target, weight in incoming_edges:
                G.add_edge(source, target, weight=weight)
    return G


def top_k_transitions_to_graph(transitions, k):
    """
    Convert raw transitions between layers into a graph structure, keeping only the top k incoming transitions
    for each cluster in layer2.

    Parameters:
    transitions: pd.Series
        A Pandas Series with a MultiIndex where each index is a tuple (layer1, layer2) indicating transitions between layers.
        Each element of the Series is a NumPy array of size (concepts_layer1, concepts_layer2) representing the transitions between clusters.
    k: int
        The number of top incoming transitions to keep for each cluster in layer2.

    Returns:
    G: nx.DiGraph
        A NetworkX directed graph where nodes are clusters and edges represent transitions with weights.
    """

    G = nx.DiGraph()

    # Iterate over transitions in reverse order
    for layer1, layer2 in reversed(transitions.index):
        transition_matrix = transitions.loc[layer1, layer2]
        # transition_matrix = transition_matrix[:-1, :-1]

        # Calculate total incoming transitions for each cluster in layer2
        total_incoming_transitions = transition_matrix.sum(axis=0)

        # Process each cluster in layer2
        for cluster2 in range(transition_matrix.shape[1]):
            # Collect all incoming edges for cluster2 with their scores
            incoming_edges = []
            for cluster1 in range(transition_matrix.shape[0]):
                transition_score = transition_matrix[cluster1, cluster2]
                if transition_score > 0:  # Only consider non-zero transitions
                    incoming_edges.append(
                        ((layer1, cluster1), (layer2, cluster2), transition_score)
                    )

            # Sort the incoming edges based on transition scores in descending order
            incoming_edges.sort(key=lambda x: x[2], reverse=True)

            # Keep only the top k incoming transitions
            top_k_incoming = incoming_edges[:k]
            top_k_incoming = filter(
                lambda x: x[2] != transition_matrix.shape[1] - 1, top_k_incoming
            )

            # Add edges to the graph
            for source, target, weight in top_k_incoming:
                G.add_edge(source, target, weight=weight)

    return G


def get_layer_ancestors(G: nx.DiGraph, n_layer: int):
    """
    Find the ancestors of the nodes in the last layer of the graph.

    Parameters
    ----------
    G: nx.DiGraph
        A NetworkX directed graph where nodes represent clusters in different layers, and edges represent transitions
        with the transition amount as an edge attribute

    Returns
    -------
    ancestors: list of nx.DiGraph
        A list of NetworkX directed graphs where nodes represent clusters in different layers, and edges represent transitions
        with the transition amount as an edge attribute. Each graph corresponds to the ancestors of a node in the last layer.
    """
    # Find all weakly connected components
    last_layer_nodes = [node for node in G.nodes() if node[0] == n_layer]
    # sort list of tuples by the second value
    last_layer_nodes = sorted(last_layer_nodes, key=lambda x: x[1])
    ancestors = [
        G.subgraph(nx.ancestors(G, node).copy().union(set([node])))
        for node in last_layer_nodes
    ]

    return ancestors
