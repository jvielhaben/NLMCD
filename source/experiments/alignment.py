import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist, pdist, squareform
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm


def convert_to_crisp(concept_activation):
    n_cluster = concept_activation.shape[1]
    cluster_assignment = concept_activation.argmax(axis=1)
    concept_activation_crisp = np.eye(n_cluster)[cluster_assignment]
    return concept_activation_crisp


def upper_triangle_cos_difference_torch(A, B, device="cuda"):
    """
    Generalized Rand Index between concept proximtiy scores with cosine distance (not used in the paper) for d_ms

    Parameters:
    A, B: np.ndarray of shape (n_samples, n_concepts) with concept proximity scores
    device: 'cuda' for GPU, 'cpu' for CPU
    """
    # Convert to torch tensors and move to the specified device (GPU or CPU)
    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)

    n_samples = A.shape[0]

    # Initialize result as a torch tensor
    result = torch.tensor(0.0, device=device)

    # Compute pairwise L1 differences using broadcasting
    for i in range(n_samples):
        # Extract the ith row and compute the difference with all subsequent rows
        Ai = A[i]
        Bi = B[i]

        # Compute differences for all pairs (i, j) where j > i
        differences_A = 1 - cosine_similarity(Ai, A[i + 1 :], dim=1)
        differences_B = 1 - cosine_similarity(Bi, B[i + 1 :], dim=1)

        # Compute the L1 difference and accumulate the result
        result += torch.abs(differences_A - differences_B).sum()

    # Move result back to CPU and convert to a scalar if necessary
    return result.item()


def upper_triangle_l1_difference_torch(A, B, device="cuda"):
    """
    Distance between concept proximtiy d_cross scores with l1-norm for d_ms (see eq. 2 in th paper).

    Parameters:
    A, B: np.ndarray of shape (n_samples, n_concepts) with concept proximity scores
    device: 'cuda' for GPU, 'cpu' for CPU
    """
    # Convert to torch tensors and move to the specified device (GPU or CPU)
    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)

    n_samples = A.shape[0]

    # Initialize result as a torch tensor
    result = torch.tensor(0.0, device=device)

    # Compute pairwise L1 differences using broadcasting
    for i in range(n_samples):
        # Extract the ith row and compute the difference with all subsequent rows
        Ai = A[i]
        Bi = B[i]

        # Compute differences for all pairs (i, j) where j > i
        differences_A = 1.0 - torch.abs(Ai - A[i + 1 :]).sum(dim=1) / 2.0
        differences_B = 1.0 - torch.abs(Bi - B[i + 1 :]).sum(dim=1) / 2.0

        # Compute the L1 difference and accumulate the result
        disc = torch.abs(differences_A - differences_B)
        assert (disc >= -0.00001).all() and (
            disc <= 1.00001
        ).all(), f"disc={disc} is not in [0,1]"
        result += disc.sum()

    # Move result back to CPU and convert to a scalar if necessary
    return result.item()


def l1_difference_torch_clusterwise(A, B, device="cuda"):
    """
    Concept-wise differences based on a decomposition of the Generalized Rand Index (see eq. 4 in the paper).

    Parameters:
    A, B: np.ndarray of shape (n_samples, n_concepts) with concept proximity scores
    device: 'cuda' for GPU, 'cpu' for CPU
    """
    # Convert to torch tensors and move to the specified device (GPU or CPU)
    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)
    n = A.size(0)

    n_samples = A.shape[0]

    # Initialize result as a torch tensor
    result = torch.zeros((A.size(1), B.size(1)), device=device)

    # TODO: process larger batches across cA, cB
    # e.g. compute difference between cA and all B at once
    # Compute pairwise L1 differences using broadcasting
    # TODO torch vmap this loop
    for cA in tqdm(range(A.size(1))):
        # for cB in range(B.size(1)):

        for i in range(n_samples):
            # TODO potentially jit this
            # Extract the ith row and compute the difference with all subsequent rows
            Ai = A[i, cA]
            # Bi = B[i,cB]
            Bi = B[i]

            # Compute differences for all pairs (i, j) where j > i
            differences_A = torch.abs(Ai - A[i + 1 :, cA])
            # differences_B = torch.abs(Bi - B[i+1:,cB])
            differences_B = torch.abs(Bi - B[i + 1 :])

            # Compute the L1 difference and accumulate the result
            # result[cA,cB] += torch.abs(differences_A - differences_B).sum(dim=0) / 2.0
            result[cA] += (
                torch.abs(differences_A.unsqueeze(1) - differences_B).sum(dim=0) / 2.0
            )

    # Move result back to CPU and convert to a scalar if necessary
    cw_diff = result.cpu().numpy() / (n * (n - 1)) * 2

    match_A, matchB = linear_sum_assignment(cw_diff)

    return cw_diff, (match_A, matchB)


### Hullermeier Rand Index ###
def hullermeier_fuzzy_rand(
    concept_activation_1, concept_activation_2, crisp=False, l1_dist=False
):
    """
    Generalized Rand Index between concept proximity scores (see eq. 3 in the paper).

    Parameters:
    concept_activation_1, concept_activation_2: np.ndarray of shape (n_samples, n_concepts) with concept proximity scores
    device: 'cuda' for GPU, 'cpu' for CPU
    """

    if crisp:
        # convert to crisp clustering
        concept_activation_1 = convert_to_crisp(concept_activation_1)
        concept_activation_2 = convert_to_crisp(concept_activation_2)

    if l1_dist:
        assert (
            concept_activation_1.flatten().min() >= 0.0
            and concept_activation_1.flatten().max() <= 1.0001
        ), "concept activations 1 are not in [0,1]"
        assert (
            concept_activation_2.flatten().min() >= 0.0
            and concept_activation_2.flatten().max() <= 1.0001
        ), "concept activations 2 are not in [0,1]"

    n = concept_activation_1.shape[0]

    if l1_dist:
        disc = upper_triangle_l1_difference_torch(
            concept_activation_1, concept_activation_2
        )
    else:
        disc = upper_triangle_cos_difference_torch(
            concept_activation_1, concept_activation_2
        )
    rand = disc / (n * (n - 1)) * 2

    return 1 - rand


### Sanity Check ###
def alignment_sanity_check_score(alignment_matrix, k=1):
    """
    Alignment sanity check based on the assumption that neighbouring layers should be maximally aligned.
    Computes the ratio of layers for which the neighbouring layer is maximally aligned.

    Parameters:
    alignment matrix: np.ndarray of shape (n_layers, n_layers) holding the alignment scores between all representations within a model.
    """
    alignment_matrix[np.equal(np.eye(alignment_matrix.shape[0]), 1)] = (
        -1
    )  # setting self-alignment to negative value
    closest_layer_diff = np.abs(
        np.diff(alignment_matrix.idxmax(axis=0).reset_index().values, axis=1)
    )
    return np.logical_and(
        np.greater(closest_layer_diff, 0), np.less_equal(closest_layer_diff, k)
    ).mean()
