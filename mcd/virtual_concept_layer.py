import os
import pickle

import torch
from torch.nn.functional import cosine_similarity
import numpy as np
from torch.linalg import pinv, vector_norm
from sklearn.metrics import pairwise_distances, mean_squared_error

try:
    import cuml.cluster.hdbscan

    HDBSCANLIB_CUML = True
except:
    import hdbscan

    HDBSCANLIB_CUML = False

try:
    from cuml.cluster import KMeans

    KMEANSLIB = "cuml"
except:
    from sklearn.cluster import KMeans

    # for GPU-accelarated version of hdbscan, only works with NVIDIA VOLTA GPU
    KMEANSLIB = "sklearn"
try:
    from cuml.decomposition import PCA

    PCALIB = "cuml"
except:
    from sklearn.decomposition import PCA

    # for GPU-accelarated version of hdbscan, only works with NVIDIA VOLTA GPU
    PCALIB = "sklearn"
try:
    from cuml import UMAP

    UMAPLIB = "cuml"
except:
    from umap import UMAP

    UMAPLIB = "umap"

import mcd.lmcd


def weighted_cluster_centroid(X, labels, probabilities, cluster_id):
    """
    From https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN.weighted_cluster_centroid

    Provide an approximate representative point for a given cluster.
    Note that this technique assumes a euclidean metric for speed of
    computation. For more general metrics use the ``weighted_cluster_medoid``
    method which is slower, but can work with the metric the model trained
    with.

    Parameters
    ----------
    cluster_id: int
        The id of the cluster to compute a centroid for.

    Returns
    -------
    centroid: array of shape (n_features,)
        A representative centroid for cluster ``cluster_id``.
    """

    if cluster_id == -1:
        raise ValueError(
            "Cannot calculate weighted centroid for -1 cluster "
            "since it is a noise cluster"
        )

    mask = labels == cluster_id
    cluster_data = X[mask]
    cluster_membership_strengths = probabilities[mask]

    return np.average(cluster_data, weights=cluster_membership_strengths, axis=0)


class StaticVCL:
    """
    Concept discovery and concept proximity scores with NLMCD, MCD, KMeans or PCA.
    """

    def __init__(self, vcl_config) -> None:
        """
        Initialize the StaticVCL instance with vcl_config that contains the following settings for concept discovery (embedding and clustering):

        name: 'umap' for NLMCD or 'ident' for other discovery methods (PCA and MCS)
        zdim: dimensionality of the UMAP embedding
        cuda: True
        cluster:
            n_cluster: number of clusters (for MCD and PCA)
            discovery:  clustering method for concept discovery: 'hdbscan' 'kmeans', 'mcd', 'pca'
            cluster_assignment_list: method(s) for computing concept proximity scores after concept discovery
                hdbscan - soft clustering with hdbscan based on distance and outlier membership
                centroid_distance - distance to cluster centroid
                projection - cosine similarity with one-dimensional linear concept subspace or maximum cosine similarity to basis of multi-dim concept subspace
                hard_clustering - as obtained by kmeans or hdbscan
            min_cluster_size: for hdbscan
            min_samples: for hdbscan
            cluster_selection_method: for hdbscan - 'leaf' or 'eom'
            metric: 'euclidean'
        umap:
            n_neighbors: controls local structure of umap embedding
            min_dist:  controls local density
            metric: 'euclidean'

        Parameters:
        - vcl_config: omegaconf configuration object containing settings for
                      embedding, clustering, and assignments described above.
        """
        self.vcl_cfg = vcl_config
        self.device = torch.device("cuda") if vcl_config.cuda else torch.device("cpu")
        self.embedder = None

    def fit(self, x):
        """
        Embed feature vectors x and perform clustering.

        Parameters:
        x: feature vectors $\phi$ from model representation, torch.tensor of shape (N, F) where N is the number of feature vectors and F is their dimensionality
        """
        self._embed(x)
        self._cluster_discovery()

    def _embed(self, x):
        """
        Embed feature vectors x.

        Parameters:
        x: feature vectors $\phi$ from model representation, torch.tensor of shape (n_samples, n_dim)
        """
        if self.vcl_cfg.name == "ident":
            embedding = x.numpy()
            self.reconstruction_loss = 0.0
        elif self.vcl_cfg.name == "pca":
            if self.embedder is None:
                self.embedder = PCA(n_components=self.vcl_cfg.zdim)
                embedding = self.embedder.fit_transform(x.numpy())
                # compute reconstruction
                self.reconstruction_loss = mean_squared_error(
                    x, self.embedder.inverse_transform(embedding)
                )
            else:
                embedding = self.embedder.transform(x.numpy())
        if self.vcl_cfg.name == "umap":
            embedder = UMAP(n_components=self.vcl_cfg.zdim, **self.vcl_cfg.umap)
            embedding = embedder.fit_transform(x.numpy())
            self.reconstruction_loss = np.nan
        self.embedded_x = embedding

    def save_clustering(self, precomputed_cluster_file):
        clustering_dict = {"clustering": self.clustering, "centroids": self.centroids}
        with open(precomputed_cluster_file, "wb") as f:
            pickle.dump(clustering_dict, f)

    def _cluster_discovery(self):
        """
        Cluster embedded feature vectors.
        """
        if (
            self.vcl_cfg.cluster.n_cluster == -1
            or self.vcl_cfg.cluster.discovery == "hdbscan"
        ):
            if HDBSCANLIB_CUML:
                hdb = cuml.cluster.hdbscan.HDBSCAN(
                    min_cluster_size=self.vcl_cfg.cluster.min_cluster_size,
                    min_samples=self.vcl_cfg.cluster.min_samples,
                    prediction_data=True,
                    cluster_selection_method=self.vcl_cfg.cluster.cluster_selection_method,
                    metric=self.vcl_cfg.cluster.metric,
                ).fit(self.embedded_x)
            else:
                hdb = hdbscan.HDBSCAN(
                    min_cluster_size=self.vcl_cfg.cluster.min_cluster_size,
                    min_samples=self.vcl_cfg.cluster.min_samples,
                    prediction_data=True,
                    cluster_selection_method=self.vcl_cfg.cluster.cluster_selection_method,
                    metric=self.vcl_cfg.cluster.metric,
                    core_dist_n_jobs=18,
                ).fit(self.embedded_x)

            self.clustering = hdb
            # for measruing noise
            self.labels = hdb.labels_
            self.noise_mask = np.greater_equal(self.labels, 0)
            self.vcl_cfg.cluster.n_cluster = int(
                np.greater_equal(np.unique(self.labels), 0).sum()
            )
            # centroid
            self.centroids = np.stack(
                [
                    weighted_cluster_centroid(
                        self.embedded_x, hdb.labels_, hdb.probabilities_, c
                    )
                    for c in range(self.vcl_cfg.cluster.n_cluster)
                ]
            )
        if self.vcl_cfg.cluster.discovery == "kmeans":
            self.clustering = KMeans(n_clusters=self.vcl_cfg.cluster.n_cluster).fit(
                self.embedded_x
            )
            self.labels = self.clustering.labels_
            self.centroids = self.clustering.cluster_centers_
        elif self.vcl_cfg.cluster.discovery == "mcd":
            cl = mcd.lmcd.ConceptSubspaces(self.embedded_x)
            cl.get_feature_array()
            cl.cluster(kmeans=True, n_clusters=self.vcl_cfg.cluster.n_cluster)
            # filter out 1 member concepts
            concepts, counts = np.unique(cl.labels, return_counts=True)
            mask = np.less_equal(counts, 2)
            cl.labels[np.where(np.isin(cl.labels, concepts[mask]))[0]] = -1
            cl.concepts = concepts[np.logical_not(mask)]
            # construct basis
            cl.conceptspace_bases(ver="FO")
            self.labels = cl.labels.astype(int)
            self.clustering = cl
            self.centroids = cl.conceptBases
        elif self.vcl_cfg.cluster.discovery == "pca":
            n_components = (
                self.vcl_cfg.cluster.n_cluster
                if self.vcl_cfg.cluster.n_cluster <= self.embedded_x.shape[1]
                else self.embedded_x.shape[1]
            )
            self.clustering = PCA(n_components=n_components)
            x_transformed = self.clustering.fit_transform(self.embedded_x)
            self.centroids = self.clustering.components_
            self.labels = x_transformed.argmax(axis=1)

    @staticmethod
    def cluster_distance(X, centroids, metric, epsilon=0.000001):
        """
        Distance between feature vectors X and cluster centroid.
        Used for concept assignment to

        Parameters:
        centroids: centroids of clusters, np.ndarray of shape (n_clusters, F)
        metric: str corresponding to metric
        """
        dist = 1 / (pairwise_distances(X, centroids, metric=metric) + epsilon)
        dist = dist / (dist.sum(axis=1, keepdims=True) + epsilon)
        return dist

    def cosine_similarity_assignment(self, X, basis):
        """
        Normalized linear projection length of feature vector onto concept directions.
        Used for proximity scores between feature vectors and concepts as linear directions.

        Parameters:
        X: np.ndarray, shape (n_samples, n_dim)
        basis: np.ndarray, shape (n_concepts, n_dim)
        """
        assignment = np.empty((X.shape[0], basis.shape[0]))
        basis = torch.tensor(basis, device=self.device)
        X = torch.tensor(X, device=self.device)
        for i in range(basis.shape[0]):
            assignment[:, i] = cosine_similarity(X, basis[i], dim=1).cpu().numpy()
        return assignment

    def grassmann_assignment(self, X, subspaceBases):
        """
        Grassmann distance between feature vectors and linear concept subspace as discovered by MCD
        - maximum normalized linear projection length of feature vector onto across basis vectors of one concept subspace.

        Parameters:
        X: feature vectors, np.ndarray of shape c
        subspaceBases: list[np.ndarray], bases of concept subspaces (as discovered by MCD).
        """
        assignment = np.empty((X.shape[0], len(subspaceBases)))
        for i, basis in enumerate(subspaceBases):
            assignment[:, i] = self.cosine_similarity_assignment(X, basis).max(axis=1)
        return assignment

    def cluster_assignment(self):
        """
        Assign concept proximity scores to embedded feature vectors.
        """
        # assignments avialable for each clustering
        available_assignments = {
            "hdbscan": ("hdbscan", "hard_clustering"),
            "mcd": ("projection",),
            "kmeans": ("centroid_distance",),
            "pca": ("projection",),
        }

        # delete non-available assignments from assignment list
        self.vcl_cfg.cluster.cluster_assignment_list = [
            a
            for a in self.vcl_cfg.cluster.cluster_assignment_list
            if a in available_assignments[self.vcl_cfg.cluster.discovery]
        ]

        self.ca = {}
        if "hdbscan" in self.vcl_cfg.cluster.cluster_assignment_list:
            print("Computing HDBSCAN assignment")
            if HDBSCANLIB_CUML:
                self.ca["hdbscan"] = cuml.cluster.hdbscan.prediction.membership_vector(
                    self.clustering, self.embedded_x
                )
            else:
                self.ca["hdbscan"] = hdbscan.prediction.membership_vector(
                    self.clustering, self.embedded_x
                )
        if "centroid_distance" in self.vcl_cfg.cluster.cluster_assignment_list:
            print("Computing centroid distance")
            self.ca["centroid_distance"] = self.cluster_distance(
                self.embedded_x, self.centroids, self.vcl_cfg.cluster.metric
            )
        if "projection" in self.vcl_cfg.cluster.cluster_assignment_list:
            print("Projecting on centroids")
            if self.vcl_cfg.cluster.discovery == "mcd":
                self.ca["projection"] = self.grassmann_assignment(
                    self.embedded_x, self.centroids
                )
            else:
                self.ca["projection"] = self.cosine_similarity_assignment(
                    self.embedded_x, self.centroids
                )
        if "hard_clustering" in self.vcl_cfg.cluster.cluster_assignment_list:
            print("Hard clustering")
            if self.vcl_cfg.cluster.discovery == "hdbscan":
                if HDBSCANLIB_CUML:
                    self.ca["hard_clustering"] = (
                        cuml.cluster.hdbscan.prediction.approximate_predict(
                            self.clustering, self.embedded_x
                        )[0]
                    )
                else:
                    self.ca["hard_clustering"] = hdbscan.prediction.approximate_predict(
                        self.clustering, self.embedded_x
                    )[0]
            elif self.vcl_cfg.cluster.discovery == "kmeans":
                self.ca["hard_clustering"] = self.ca = self.clustering.predict(
                    self.embedded_x
                )

    def save_assignment(self, save_dir, train=False):
        for k in self.ca:
            pre = "train_" if train else ""
            np.save(os.path.join(save_dir, f"{pre}{k}-clustering.npy"), self.ca[k])
