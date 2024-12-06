import timm

from source.data import imagenet


def load_data(
    config,
    batch_size: int,
    train: bool = False,
    numpy: bool = True,
    return_label: bool = True,
    shuffle: bool = False,
    cuda: bool = True,
):
    if config.name == "imagenet":
        loader = imagenet.imagenet_loader(
            config, batch_size, train, return_label, numpy, shuffle, cuda
        )

    return loader


def load_model(config):
    if config.name == "imagenet":
        model_ckpt = config.params.representation_model_ckpt
        model = timm.create_model(model_ckpt, pretrained=True)
        model = model.eval()
    return model
