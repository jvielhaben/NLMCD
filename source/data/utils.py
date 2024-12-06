import torch.nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import random


#### Dataset aux ####
class ToTensor(torch.nn.Module):
    def __init__(self, unsqueeze) -> None:
        super().__init__()
        self.unsqueeze = unsqueeze

    def forward(self, x):
        x = transforms.functional.to_tensor(x)
        if self.unsqueeze:
            x = x.unsqueeze(0)
        return x


class Unsqueeze(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(0)


class Squeeze(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x.squeeze(0)


class ToCUDA(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x.cuda()


class PatchDivide(torch.nn.Module):
    def __init__(
        self, patch_size=7, image_size=28, n_channels=1, batchwise=False
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size
        self.n_channels = n_channels
        self.batchwise = batchwise

    def forward(self, x):
        x = x.squeeze()
        if self.batchwise:
            patches = torch.zeros(
                x.shape[0],
                self.num_patches,
                self.num_patches,
                self.patch_size,
                self.patch_size,
                self.n_channels,
            )
            if self.n_channels > 1:
                x = x.permute(0, 2, 3, 1)
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch = x[
                        :,
                        i * self.patch_size : (i + 1) * self.patch_size,
                        j * self.patch_size : (j + 1) * self.patch_size,
                    ]
                    patches[:, i, j] = patch
            patches = patches.view(
                -1,
                self.num_patches**2,
                self.patch_size,
                self.patch_size,
                self.n_channels,
            ).squeeze()
            patches = patches.permute(0, 1, 4, 2, 3)  # original dimension order
        else:
            patches = torch.zeros(
                self.num_patches,
                self.num_patches,
                self.patch_size,
                self.patch_size,
                self.n_channels,
            )
            if self.n_channels > 1:
                x = x.permute(1, 2, 0)
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch = x[
                        i * self.patch_size : (i + 1) * self.patch_size,
                        j * self.patch_size : (j + 1) * self.patch_size,
                    ]
                    # print(patches.size(), patch.size())
                    patches[i, j] = patch
            patches = patches.view(
                self.num_patches**2, self.patch_size, self.patch_size, self.n_channels
            )
            patches = patches.permute(0, 3, 1, 2).squeeze()  # original dimension order

        return patches


class ToNumpy(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x):
        return x.numpy()


class DimensionTransformer(torch.nn.Module):
    """
    Reorganize tensor dimensions.

    Example: Strategy '0,1,2,3->01,23': Reorganize the dimensions of a tensor of shape $(batchsize, channel, h, w)$ to $(batchsize*channel,h*w)$.
    """

    def __init__(self, transformation, original_shape=None):
        super().__init__()
        self.transformation = transformation
        if original_shape is None:
            self.shape_is_set = False
        else:
            self.original_shape = original_shape
            self.shape_is_set = True
        self.mode = "forward"

        # necessary for collate function
        batch_dimensions = list(
            map(int, list(self.transformation.split("->")[1].split(",")[0]))
        )
        batch_dimensions.remove(0)
        self.batch_dimensions = batch_dimensions

        self.n_called = 0

    def _get_transformed_shape(self):
        original_dimensions, transformed_dimensions = self.transformation.split("->")
        assert len(original_dimensions.split(",")) == len(
            self.original_shape
        ), f"{original_dimensions} vs. {self.original_shape}"

        transformed_dimension_perumtation = tuple(
            map(int, list(transformed_dimensions.replace(",", "")))
        )
        assert len(transformed_dimension_perumtation) == len(self.original_shape)

        transformed_shape = []
        for dim in transformed_dimensions.split(","):
            if self.mode == "forward":
                transformed_dim = 1
                for d in dim:
                    transformed_dim *= self.original_shape[int(d)]
                transformed_shape.append(transformed_dim)
            elif self.mode == "backward":
                for d in dim:
                    transformed_shape.append(self.original_shape[int(d)])

        # make 0th dim with batch size variable
        transformed_shape = (-1,) + tuple(transformed_shape)[1:]

        return transformed_dimension_perumtation, transformed_shape

    def forward_mode(self):
        self.mode = "forward"

    def backward_mode(self):
        self.mode = "backward"

    def _forward(self, x):
        if not self.shape_is_set:
            # set it the first time forward is called:
            self.original_shape = tuple(x.shape)
            self.shape_is_set = True
        assert (
            tuple(x.size())[1:] == self.original_shape[1:]
        ), f"{x.size()}, {self.original_shape}"
        transformed_dimension_perumtation, transformed_shape = (
            self._get_transformed_shape()
        )

        # Reshape the tensor based on the transformation
        transformed_tensor = (
            x.permute(*transformed_dimension_perumtation)
            .contiguous()
            .view(transformed_shape)
        )

        return transformed_tensor

    def _backward(self, transformed_tensor):
        transformed_dimension_perumtation, transformed_shape = (
            self._get_transformed_shape()
        )
        original_tensor = transformed_tensor.view(transformed_shape)
        original_tensor = original_tensor.permute(
            *transformed_dimension_perumtation
        ).contiguous()
        return original_tensor

    def forward(self, x):
        if self.mode == "forward":
            return self._forward(x)
        elif self.mode == "backward":
            return self._backward(x)

    def _test(self, x):
        _x_copy = x.data.clone()
        assert torch.all(_x_copy == self._backward(self._forward(x)))


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


##### Dataloader aux #####
def numpy_collate_fn(batch):
    # 'batch' is a list of tuples, where each tuple contains the data and label for one sample
    # Assuming that each sample is represented as (data, label)

    # Separate the data and labels from the batch
    data_batch, label_batch = zip(*batch)

    # Convert data and labels to numpy arrays
    data_np = np.array(data_batch)
    label_np = np.array(label_batch)

    return data_np, label_np


def create_symmetric_weight_matrix(size=14):
    """
    For center-focused sampling of tokens
    """
    center = size / 2  # Center for even size
    x = np.arange(size)
    y = np.arange(size)
    x, y = np.meshgrid(x, y)

    # Calculate the distance from the center
    distance = np.sqrt((x - (center - 0.5)) ** 2 + (y - (center - 0.5)) ** 2)

    # Gaussian weights based on distance
    sigma = size / 10  # Adjust this to control the spread
    weights = np.exp(-(distance**2) / (2 * sigma**2))

    # Normalize to sum to 1
    weights /= np.sum(weights)

    return weights.flatten()


def dimension_trafo_collate_fn(
    batch,
    dimension_transformer,
    dimension_transformer_idx,
    subsample_ratio=1.0,
    center_sampling=False,
    numpy=False,
):
    random.seed(42)
    np.random.seed(42)

    # Separate the data and labels from the batch
    data_batch, label_batch = zip(*batch)
    data_batch = torch.stack(data_batch)
    label_batch = (
        torch.stack(label_batch)
        if torch.is_tensor(label_batch[0])
        else torch.tensor(label_batch)
    )
    # print(data_batch.shape, label_batch.shape)

    weights = (
        create_symmetric_weight_matrix(size=int(data_batch.shape[1] ** 0.5))
        if center_sampling
        else None
    )

    if subsample_ratio < 1.0:
        n = data_batch.shape[1]
        n_remain = int(n * subsample_ratio)
        idx = np.array(random.choices(np.arange(n), k=n_remain, weights=weights))
        data_batch = data_batch[:, idx]
        idx_batch = torch.from_numpy(
            np.repeat(
                idx[np.newaxis, :, np.newaxis], axis=0, repeats=data_batch.shape[0]
            )
        )
    else:
        idx_batch = None

    return dimension_collator(
        data_batch,
        idx_batch,
        label_batch,
        dimension_transformer,
        dimension_transformer_idx,
        numpy,
    )


def dimension_collator_label(
    label_batch, data_batch_shape, dimension_transformer, numpy=False
):
    # for transforming label batch
    batch_dimensions = [
        data_batch_shape[i] for i in dimension_transformer.batch_dimensions
    ]
    label_batch_shape_original = label_batch.shape

    batch_dimensions_prod = np.prod(batch_dimensions)

    # transform the dimension of label by expanding if batch size has increased
    assert data_batch_shape[0] >= len(
        label_batch
    ), "len(data_batch)<len(label_batch) not allowed"
    label_batch = label_batch.view(
        label_batch_shape_original[0],
        *[1] * len(dimension_transformer.batch_dimensions),
        *label_batch_shape_original[1:],
    )
    label_batch = label_batch.expand(
        -1, *batch_dimensions, *[-1] * len(label_batch_shape_original[1:])
    )
    label_batch = label_batch.reshape(
        label_batch_shape_original[0] * batch_dimensions_prod,
        *label_batch_shape_original[1:],
    )

    if numpy:
        # Convert labels to numpy arrays
        label_batch = np.array(label_batch)

    return label_batch


def dimension_collator(
    data_batch,
    idx_batch,
    label_batch,
    dimension_transformer,
    dimension_transformer_idx,
    numpy=False,
):
    data_batch_shape_original = data_batch.shape  # for label barch collation

    # transform dimensions of data
    data_batch = dimension_transformer(data_batch)
    if idx_batch is not None:
        idx_batch = dimension_transformer_idx(idx_batch).squeeze()

    if numpy:
        # Convert data to numpy arrays
        data_batch = data_batch.numpy()
        if idx_batch is not None:
            idx_batch = idx_batch.numpy()

    label_batch = dimension_collator_label(
        label_batch, data_batch_shape_original, dimension_transformer, numpy
    )

    return data_batch, idx_batch, label_batch


def preload_dataset(dataloader, batch_size, load_data=True, return_loader=True):
    batch = next(iter(dataloader))
    if len(batch) == 3:
        data_shape, idx, label_shape = batch[0].shape, batch[1], batch[2].shape
    else:
        data_shape, label_shape = batch[0].shape, batch[1].shape
        idx = None
    print(data_shape[0], len(dataloader))
    N_estimate = data_shape[0] * len(
        dataloader
    )  # data_shape[0] can be different from batch size because of collate

    print("Estimated number of samples", N_estimate)
    print("PRELOADING DATASET")
    if load_data:
        X_all = torch.empty((N_estimate, *data_shape[1:]))
        if idx is not None:
            idx_all = torch.empty((N_estimate, *idx.shape[1:]))
    else:
        X_all = None
    y_all = torch.empty((N_estimate, *label_shape[1:]))
    c = 0
    # for batch in tqdm(dataloader):
    for batch in dataloader:
        if len(batch) == 3:
            X, idx, y = batch
        else:
            X, y = batch
        N_i = X.shape[0]
        if load_data:
            X_all[c : c + N_i] = X
            if idx is not None:
                idx_all[c : c + N_i] = idx
        y_all[c : c + N_i] = y
        c += N_i
    # in case last batch is smaller than batch size
    if load_data:
        X_all = X_all[:c]
        if idx is not None:
            idx_all = idx_all[:c]
    y_all = y_all[:c]

    if not return_loader:
        return X_all, idx_all, y_all

    if idx is not None:
        loaded_dataset = TensorDataset(X_all, idx_all, y_all)
    else:
        loaded_dataset = TensorDataset(X_all, y_all)

    if batch_size is None:
        batch_size = len(loaded_dataset)

    return DataLoader(
        loaded_dataset, batch_size=batch_size, num_workers=dataloader.num_workers
    )  # , sampler=dataloader.sampler)


def get_patch_labels(n_samples, label_type="cls", y_gt=None, n_patches=197):
    """
    imagenet ViT patch labels
    """
    if label_type == "sample":
        label = np.repeat(
            np.arange(n_samples)[:, np.newaxis], repeats=n_patches, axis=1
        )
        label_transformer = DimensionTransformer(transformation="0,1->01")
        label = label_transformer(torch.from_numpy(label))
    elif label_type == "gt":
        label = np.repeat(y_gt, n_patches)
    elif label_type == "cls":
        label = torch.zeros(n_samples, n_patches)
        label[:, 0] = 1
        label_transformer = DimensionTransformer(transformation="0,1->01")
        label = label_transformer(label)
    elif label_type == "loc":
        label = np.repeat(np.arange(n_patches)[np.newaxis], repeats=n_samples, axis=0)
        label_transformer = DimensionTransformer(transformation="0,1->01")
        label = label_transformer(torch.from_numpy(label))

    return label
