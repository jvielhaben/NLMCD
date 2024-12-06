import os
import dash
import dash_cytoscape as cyto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from dash import html
from matplotlib import patches

from source.experiments.cfg_utils import (
    imagenet_loader,
    threshold_assignment,
    compute_raw_transition_scores,
    get_trafos_in,
    create_dataset,
    single_limit_transitions_to_graph,
    get_layer_ancestors,
)
from source.data.imagenet import CustomImageNet
from source.experiments.eval_utils import load_concept_activations, load_configs_df
from torchvision.transforms import Normalize


def draw_patch_frame(img, index, ax, kernel_size=4, patch_size=16, stride=1):
    # Calculate the number of patches in the image
    n_patches_x = (img.shape[1] // patch_size) // stride - kernel_size + 1
    patch_indices = np.arange(n_patches_x**2).reshape(n_patches_x, n_patches_x)

    # Calculate the row and column of the patch
    row, col = np.where(index == patch_indices)

    # Calculate the top-left corner of the patch in the original image
    top_left_x = col * stride * patch_size
    top_left_y = row * stride * patch_size

    ax.imshow(img)

    # Create a rectangle patch (kernel size)
    rect = patches.Rectangle(
        (top_left_x, top_left_y),
        patch_size * kernel_size,
        patch_size * kernel_size,
        linewidth=0.1,
        edgecolor="yellow",
        facecolor="none",
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)


def save_node_image(fig, layer, concept, output_dir):
    """
    Save the figure to an image file.
    """
    image_path = os.path.join(output_dir, f"{model}_node_{layer}_{concept}.png")
    fig.savefig(image_path, dpi=DPI)
    plt.close(fig)
    return f"{model}_node_{layer}_{concept}.png"


def visualize_cluster(
    cluster_idx,
    model,
    soft_clustering,
    hard_clustering,
    token_idx,
    sample_idx,
    cfg_data,
    n_patches,
    ref_cluster_idx=None,
    select_from_all_samples=False,
    n_samples=6,
    random=True,
    title=False,
):
    print("Visualize node")
    if ref_cluster_idx is not None:
        mask = np.logical_and(
            hard_clustering == cluster_idx,
            soft_clustering.argmax(axis=1) == ref_cluster_idx,
        )
    else:
        mask = hard_clustering == cluster_idx
    if select_from_all_samples:
        mask = np.ones(soft_clustering.shape[0]) == 1
    soft_clustering = soft_clustering[mask]
    if random:
        idx = np.random.choice(soft_clustering.shape[0], size=n_samples, replace=False)
    else:
        idx = soft_clustering.max(axis=1).argsort()[-n_samples:]
    if select_from_all_samples:
        idx = soft_clustering[:, cluster_idx].argsort()[-n_samples:]

    sample_idx = sample_idx[mask][idx][:n_samples]
    # token_idx = token_idx[idx]

    loader = imagenet_loader(
        cfg_data,
        model,
        dataset,
        batch_size=n_samples,
        train=True,
        return_label=False,
        cuda=False,
        indices_subsample=sample_idx,
    )
    input_images, y = next(iter(loader))

    # undo normalization
    norm = loader.dataset.dataset.transform.transforms[-1]
    norm_inverse = Normalize(mean=-norm.mean / norm.std, std=1 / norm.std)
    input_images = norm_inverse(input_images).permute(0, 2, 3, 1).float()
    input_images = torch.clip(input_images, 0, 1)

    n_rows = n_samples // 3
    fig, ax = plt.subplots(
        n_rows, 3, figsize=(0.12, 0.03926 * n_rows + (n_rows - 1) * 0.001), dpi=DPI
    )
    ax = ax.flatten()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

    kernel_size = cfg_data.params.kernel_size if cfg_data.params.pool_token else 1

    for i, idx_i in enumerate(idx):
        # ax[i].imshow(img)
        # img_idx = (y==sample_idx[i]).argmax()   #What was that about????? Why did it still work with this on?
        draw_patch_frame(
            input_images[i], token_idx[idx_i], ax[i], kernel_size=kernel_size
        )
        if title:
            ax[i].set_title(
                f"{soft_clustering[idx_i].max():.3f}, {soft_clustering[idx_i].argmax()}",
                fontsize=10,
            )
        ax[i].axis("off")

    return fig


# Convert the NetworkX graph to Dash Cytoscape format with manual positioning
def nx_to_cytoscape_data_with_positions(G, output_dir):
    elements = []

    layer_dict = {}
    for node in G.nodes:
        layer = node[0]
        if layer not in layer_dict:
            layer_dict[layer] = []
        layer_dict[layer].append(node)

    # Now manually position nodes based on their layers
    x_spacing = 200
    y_spacing = 150

    for layer, nodes in layer_dict.items():
        # Center nodes vertically in the layer
        layer_center_y = y_spacing * len(G.nodes) / 2
        layer_height = y_spacing * (len(nodes) - 1)
        y_start = layer_center_y - layer_height / 2

        for i, node in enumerate(nodes):
            y_position = y_start + i * y_spacing

            # Generate image for each node
            fl = layer
            fig = visualize_cluster(
                node[1],
                model_loaded,
                soft_assignments[("hdbscan", model, discovery, fl)],
                hard_assignments[("hard_clustering", model, discovery, fl)],
                token_idx=token_idx_dict[("", model, discovery, 1)],
                sample_idx=sample_idx,
                n_samples=6,
                cfg_data=cfg_data,
                n_patches=121,
            )

            # Save the figure as an image
            image_path = save_node_image(fig, node[0], node[1], output_dir)

            # Use the image as the background for the node
            elements.append(
                {
                    "data": {"id": str(node), "label": str(node)},
                    "position": {"x": layer * x_spacing, "y": y_position},
                    "style": {
                        "background-image": f"/assets/{image_path}",  # Use the image as the node background
                        "background-opacity": 0,  # Make the node's default shape transparent
                        "background-fit": "cover",  # Ensure the image covers the node fully
                        "shape": "rectangle",  # Use a rectangle shape to fit the image
                        "border-width": 0,  # Remove any border
                        "width": "162px",  # Set desired width
                        "height": "115px",  # Set desired height
                    },
                }
            )

    # Add edges with transition scores as labels
    for edge in G.edges:
        transition_score = G.edges[edge][
            "weight"
        ]  # Extract the transition score for the edge
        elements.append(
            {
                "data": {
                    "source": str(edge[0]),
                    "target": str(edge[1]),
                    "weight": transition_score,
                    "label": str(
                        transition_score
                    ),  # Set the transition score as the edge label
                },
                "style": {
                    "label": str(transition_score),  # Display the transition score
                    "font-size": "12px",  # Adjust font size for visibility
                    "text-rotation": "autorotate",  # Rotate text to align with the edge
                    "text-background-color": "white",  # Background color for text (optional)
                    "text-background-opacity": 0.8,  # Background opacity for readability
                    "text-background-padding": "3px",  # Padding around the text
                },
            }
        )

    return elements


DPI = 5000


exp_dir = "/data1/bareeva/nlmcd/run_1_umap"
artifacts_dir = "../wandb_export"

out_dir = "/data1/bareeva/nlmcd/output/"
concepts_dir = "./assets"
local_dir = "./assets"

date = "2024-10-11_15-00-00"
measured = ["n_cluster", "n_noise", "n_sample"]

models = [
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_224.dino",
    "vit_base_patch16_clip_224.openai",
]

discovery = "umap_hdbscan_50-20_1-0.01-0.25"

assignment_types = [
    {"hard": True, "noise_threshold": None},
    {"hard": False, "noise_threshold": 0.5},
    {"hard": False, "noise_threshold": 0.7},
]


model = "vit_base_patch16_clip_224.openai"
assignment_type = {"hard": True, "noise_threshold": None}
hard = assignment_type["hard"]
noise_threshold = assignment_type["noise_threshold"]


# load config data
df = load_configs_df(exp_dir, start_date=date, measured=measured)
df["noise_ratio"] = df["n_noise"] / df["n_sample"]
df["n_cluster"] = df["n_cluster"].astype(int)
df["dataset.params.feature_layer"] = df["dataset.params.feature_layer"].astype(int)
df["config_path_min"] = df["config_path"].apply(
    lambda cpath: os.path.join(*cpath.split("/")[-2:])
)
df = df.sort_values("now_dir", ascending=True)

df["run_id"] = df["config_path"].apply(lambda p: p.split("/")[5]).astype(int)
df.set_index("run_id", inplace=True)

# load assignments

min_run_number = 1473274
max_run_number = 1473321

df_sub = df.sort_index().loc[min_run_number:max_run_number]

indexers = [
    "dataset.params.representation_model_ckpt",
    "vcl",
    "dataset.params.feature_layer",
]

# set index
df_sub = df_sub.reset_index().set_index(indexers)

soft_assignments = load_concept_activations(
    df_sub,
    exp_dir,
    train=True,
    cluster_assignment="hdbscan",
    filename_root="clustering.npy",
    take_parent=False,
)

hard_assignments = load_concept_activations(
    df_sub,
    exp_dir,
    train=True,
    cluster_assignment="hard_clustering",
    filename_root="clustering.npy",
    take_parent=False,
)

assignments = {}
assignments.update(soft_assignments)
assignments.update(hard_assignments)


# turn assignments into a df for convenience
index = pd.MultiIndex.from_tuples(assignments.keys())
values = list(assignments.values())
values = [(v,) for v in values]
assignments = pd.DataFrame(
    index=index, data=values, columns=["assignment"], dtype="object"
)
assignments = assignments.unstack(level=0).droplevel(0, axis=1)

# compute transition scores
print(f"Computing transitions for {model} {discovery} {assignment_type}")

if hard:
    assignment = "hard_clustering"
    finalize_assignment = lambda x: x[:, np.newaxis]
else:
    assignment = "hdbscan"
    finalize_assignment = lambda x: threshold_assignment(x, noise_threshold, hard)

assignments_select = assignments.loc[model, discovery][assignment]

assignments_select = assignments_select.loc[
    assignments_select.apply(lambda x: x is not None)
]
available_fl = assignments_select.index.unique()

index = pd.MultiIndex.from_tuples(
    [(available_fl[i], available_fl[i + 1]) for i in range(len(available_fl[:-1]))]
)

# calculate transition scores
transition_scores = pd.Series(index=index, dtype="object")
raw_transitions = pd.Series(index=index, dtype="object")
final_assignments = [finalize_assignment(assignments_select.loc[available_fl[0]])]

for i, fl in enumerate(available_fl[:-1]):
    fl1 = available_fl[i]
    fl2 = available_fl[i + 1]
    final_assignments.append(finalize_assignment(assignments_select.loc[fl2]))
    raw_transitions.loc[fl1, fl2] = compute_raw_transition_scores(
        final_assignments[-2], final_assignments[-1]
    )

raw_transitions = raw_transitions.iloc[6:9]

# load model and data
df_sub_indices = {
    "vit_base_patch16_224.augreg_in1k": 35,
    "vit_base_patch16_224.dino": 0,
    "vit_base_patch16_clip_224.openai": 23,
}

cfg_data = df_sub.iloc[df_sub_indices[model]]["config"].dataset
cfg_data.params.root = "/data1/datapool/ImageNet/ILSVRC/Data/CLS-LOC/"
cfg_data.params.feature_layer = 0

model_loaded = timm.create_model(
    cfg_data.params.representation_model_ckpt, pretrained=True
)
model_loaded = model_loaded.eval()

transforms, dimension_trafo_collate = get_trafos_in(cfg_data, model_loaded, cuda=True)

root = cfg_data.params.root
dataset = CustomImageNet(root, split="train", transform=transforms)

# get samples idx from dataset creation
subset_dataset, _ = create_dataset(
    cfg_data,
    model_loaded,
    dataset,
    return_label=True,
    cuda=True,
    train=True,
    indices_subsample=None,
)
sample_idx = subset_dataset.indices

# repeat as often as token were selected from one image
if int(cfg_data.subsample_ratio * 121) > 1:
    sample_idx = np.repeat(sample_idx, repeats=int(cfg_data.subsample_ratio * 196 / 49))

cluster_idx = 0
token_idx_dict = load_concept_activations(
    df_sub.iloc[[df_sub_indices[model]]],
    exp_dir,
    train=False,
    cluster_assignment="",
    filename_root="token_idx.npy",
    take_parent=False,
)

nx_graph = single_limit_transitions_to_graph(raw_transitions, 0.05)


# Concept Graph Formation

# select layer and node
LAYER = 10
N_NODE = 434
ancestors = get_layer_ancestors(nx_graph, n_layer=LAYER)
# sorted_ancestors = sorted(ancestors, key=lambda s: len(s.nodes), reverse=True)
nx_graph = ancestors[N_NODE]

# create assets folder if doesn't exist
if not os.path.exists(concepts_dir):
    os.makedirs(concepts_dir)

# generate the elements for Dash Cytoscape
np.random.seed(43)
cyto_elements = nx_to_cytoscape_data_with_positions(nx_graph, output_dir=concepts_dir)

app = dash.Dash(__name__)
# Define the app layout
app.layout = html.Div(
    [
        cyto.Cytoscape(
            id="cytoscape",
            elements=cyto_elements,
            style={"width": "100%", "height": "800px"},
            layout={
                "name": "preset"
            },  # Use the preset layout to maintain manual positions
        )
    ]
)


if __name__ == "__main__":
    app.run(debug=False)
