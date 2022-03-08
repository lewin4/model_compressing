# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, List, NewType, Optional, Set, Union
from collections import OrderedDict
import torch

from ..compressed_layers.CompressedConv2d import CompressedConv2d
from ..compressed_layers.CompressedConvTranspose2d import CompressedConvTranspose2d
from ..compressed_layers.CompressedLinear import CompressedLinear
from ..compression.coding import get_kmeans_fn, get_num_centroids

RecursiveReplaceFn = NewType("RecursieReplaceFn", Callable[[torch.nn.Module, torch.nn.Module, int, str, str], bool])


def _replace_child(model: torch.nn.Module, child_name: str, compressed_child_model: torch.nn.Module, idx: int) -> None:
    """Replaces a given module into `model` with another module `compressed_child_model`

    Parameters:
        model: Model where we are replacing elements
        child_name: The key of `compressed_child_model` in the parent `model`. Used if `model` is a torch.nn.ModuleDict
        compressed_child_model: Child module to replace into `model`
        idx: The index of `compressed_child_model` in the parent `model` Used if `model` is a torch.nn.Sequential
    """
    if isinstance(model, torch.nn.Sequential):
        # Add back in the correct position
        model[idx] = compressed_child_model
    elif isinstance(model, torch.nn.ModuleDict):
        model[child_name] = compressed_child_model
    else:
        model.add_module(child_name, compressed_child_model)


def prefix_name_lambda(prefix: str) -> Callable[[str], str]:
    """Returns a function that preprends `prefix.` to its arguments.

    Parameters:
        prefix: The prefix that the return function will prepend to its inputs
    Returns:
        A function that takes as input a string and prepends `prefix.` to it
    """
    return lambda name: (prefix + "." + name) if prefix else name


@torch.no_grad()
def apply_recursively_to_model(fn: RecursiveReplaceFn, model: torch.nn.Module, prefix: str = "") -> None:
    """Recursively apply fn on all modules in models

    Parameters:
        fn: The callback function, it is given the parents, the children, the index of the children,
            the name of the children, and the prefixed name of the children
            It must return a boolean to determine whether we should stop recursing the branch
        model: The model we want to recursively apply fn to
        prefix: String to build the full name of the model's children (eg `layer1` in `layer1.conv1`)
    """
    get_prefixed_name = prefix_name_lambda(prefix)

    for idx, named_child in enumerate(model.named_children()):

        child_name, child = named_child
        child_prefixed_name = get_prefixed_name(child_name)

        if fn(model, child, idx, child_name, child_prefixed_name):
            continue
        else:
            apply_recursively_to_model(fn, child, child_prefixed_name)


layer_list = OrderedDict()


@torch.no_grad()
def apply_recursively_to_model_special(
        fn: RecursiveReplaceFn,
        model: torch.nn.Module,
        prefix: str = ""
) -> None:
    """Recursively apply fn on all modules in models

    Parameters:
        fn: The callback function, it is given the parents, the children, the index of the children,
            the name of the children, and the prefixed name of the children
            It must return a boolean to determine whether we should stop recursing the branch
        model: The model we want to recursively apply fn to
        prefix: String to build the full name of the model's children (eg `layer1` in `layer1.conv1`)
    """
    get_prefixed_name = prefix_name_lambda(prefix)
    global layer_list
    for idx, named_child in enumerate(model.named_children()):

        child_name, child = named_child
        child_prefixed_name = get_prefixed_name(child_name)

        if fn(model, child, idx, child_name, child_prefixed_name):
            layer_list[child_prefixed_name] = (model, child, idx, child_name)
            # if len(layer_list) == 5:
            #     multi_compression(model, layer_list, compression_config)
            continue
        else:
            apply_recursively_to_model_special(fn, child, prefix=child_prefixed_name)


def get_code_and_codebook(
        reshaped_layers_weight: OrderedDict,
        multi_layer_specs: Dict,
        n_multi_layer: int
) -> Dict:
    """生成多层共享的编码和码本

    Parameters:
        reshaped_layers_weight: 这是一个有序字典，键-每个卷积层的在整个模型中的名字，值-(整形后的层权重
        值，在上一层中的序号(假如上一层是一个有序module的话))
        k: 码本的长度或者说是大小
        kmeans_n_iters: 量化次数
    Returns:
        一个字典，键-卷积核大小，值-(生成的码本, (每个卷积层的在整个模型中的名字, 生成的该层的编码))
    """

    # group_list = []
    # a_dict = {}
    # for name, content in reshaped_layers_weight.items():
    #     a_dict[name] = content
    #     if len(a_dict) >= n_multi_layer:
    #         group_list.append(a_dict)
    #         a_dict = {}
    # for i in range(0, n_multi_layer):

    weight_groups = {}
    for i, (name, content) in enumerate(reshaped_layers_weight.items()):
        reshaped_weight = content[0]
        id = content[1]
        parent = content[2]
        assert len(reshaped_weight.shape) == 2
        size_code, size = reshaped_weight.size()

        if size not in weight_groups:
            weight_groups[size] = [[torch.Tensor().to(reshaped_weight.device), []]]
        elif len(weight_groups[size][-1][1]) % n_multi_layer == 0:
            weight_groups[size].append([torch.Tensor().to(reshaped_weight.device), []])

        weight_groups[size][-1][0] = torch.cat((weight_groups[size][-1][0], reshaped_weight), 0)
        weight_groups[size][-1][1].append([name, id, parent, size_code, int(size_code/content[3])])
        # if len(weight_groups[size][-1][1]) % n_multi_layer == 0:
        #     weight_groups[size].append([torch.Tensor().to(reshaped_weight.device), []])
    #     del reshaped_weight
    #     del parent
    # del reshaped_layers_weight

    # num_blocks_per_row = (c_in * kernel_height * kernel_width) // subvector_size
    # num_centroids = get_num_centroids(num_blocks_per_row, c_out, k)

    group_codebook_and_codes = {"codebook": {}, "layer_code": []}
    for size, contents in weight_groups.items():

        for i, content in enumerate(contents):
            training_set = content[0]
            kmeans_fn = get_kmeans_fn(multi_layer_specs[size].get("k_means_type"))
            k = multi_layer_specs[size].get("k")
            kmeans_n_iters = multi_layer_specs[size].get("k_means_n_iters")
            codebook, codes = kmeans_fn(training_set, k=k, n_iters=kmeans_n_iters)
            # del training_set
            last = 0
            for j, layer in enumerate(content[1]):
                currnt_code = codes[last:layer[3]]
                currnt_code = currnt_code.view(-1, layer[4])
                layer.append(currnt_code)
                layer.append("codebook.size"+str(size)+"."+str(i))
                last = layer[3]
                content[1][j] = layer

            group_codebook_and_codes["codebook"]["codebook.size"+str(size)+"."+str(i)] = codebook
            group_codebook_and_codes["layer_code"] = group_codebook_and_codes["layer_code"] + content[1]
    return group_codebook_and_codes

def get_reshapeed_weight(
        conv: torch.nn.Conv2d,
        large_subvectors: bool,
        pw_subvector_size: int,
) -> torch.Tensor:
    weight = conv.weight.detach()

    c_out, c_in, kernel_width, kernel_height = weight.size()
    reshaped_weight = weight.reshape(c_out, -1).detach()

    # For 1x1 convs, this is always 1
    subvector_size = kernel_height * kernel_width
    is_pointwise_convolution = subvector_size == 1

    # Determine subvector_size
    if is_pointwise_convolution:
        subvector_size = pw_subvector_size
    if large_subvectors and not is_pointwise_convolution:
        subvector_size *= 2

    assert (c_in * kernel_height * kernel_width) % subvector_size == 0

    training_set = reshaped_weight.reshape(-1, subvector_size)
    return training_set


def compress_model(
        model: torch.nn.Module,
        ignored_modules: Union[List[str], Set[str]],
        k: int,
        k_means_n_iters: int,
        k_means_type: str,
        fc_subvector_size: int,
        pw_subvector_size: int,
        large_subvectors: bool,
        layer_specs: Optional[Dict] = None,
        multi_layer_specs: Optional[Dict] = None,
        n_multi_layer: int = 5
) -> torch.nn.Module:
    """
    Given a neural network, modify it to its compressed representation with hard codes
      - Linear is replaced with compressed_layers.CompressedLinear
      - Conv2d is replaced with compressed_layers.CompressedConv2d
      - ConvTranspose2d is replaced with compressed_layers.CompressedConvTranspose2d

    Parameters:
        model: Network to compress. This will be modified in-place
        ignored_modules: List or set of submodules that should not be compressed
        k: Number of centroids to use for each compressed codebook
        k_means_n_iters: Number of iterations of k means to run on each compressed module
            during initialization
        k_means_type: k means type (kmeans, src)
        fc_subvector_size: Subvector size to use for linear layers
        pw_subvector_size: Subvector size for point-wise convolutions
        large_subvectors: Kernel size of K^2 of 2K^2 for conv layers
        layer_specs: Dict with different configurations for individual layers
    Returns:
        The passed model, which is now compressed
    """
    if layer_specs is None:
        layer_specs = {}

    if multi_layer_specs is None:
        multi_layer_specs = {}

    def multi_compression(
            model: torch.nn.Module,
    ):


        reshaped_layers_weight = OrderedDict()
        global layer_list
        for name, content in layer_list.items():
            parent = content[0]
            layer = content[1]
            c_out, c_in,  kernel_width, kernel_height = layer.weight.shape
            id_in_parent = content[2]
            reshaped_weight = get_reshapeed_weight(layer, large_subvectors, pw_subvector_size)
            reshaped_layers_weight[name] = (reshaped_weight, id_in_parent, parent, c_out, c_in)
        # del layer_list
        group_codebook_and_codes = get_code_and_codebook(reshaped_layers_weight, multi_layer_specs, n_multi_layer)
        # return group_codebook_and_codes


    def _compress_and_replace_layer(
            parent: torch.nn.Module, child: torch.nn.Module, idx: int, name: str, prefixed_child_name: str
    ) -> bool:
        """Compresses the `child` layer and replaces the uncompressed version into `parent`"""

        assert isinstance(parent, torch.nn.Module)
        assert isinstance(child, torch.nn.Module)

        if prefixed_child_name in ignored_modules:
            return False

        child_layer_specs = layer_specs.get(prefixed_child_name, {})

        _k = child_layer_specs.get("k", k)
        _kmeans_n_iters = child_layer_specs.get("kmeans_n_iters", k_means_n_iters)
        _kmeans_fn = get_kmeans_fn(child_layer_specs.get("kmeans_type", k_means_type))
        _fc_subvector_size = child_layer_specs.get("subvector_size", fc_subvector_size)
        _large_subvectors = child_layer_specs.get("large_subvectors", large_subvectors)
        _pw_subvector_size = child_layer_specs.get("subvector_size", pw_subvector_size)

        if isinstance(child, torch.nn.Conv2d):
            # compressed_child = CompressedConv2d.from_uncompressed(child, _k, _kmeans_n_iters, _kmeans_fn,
            #                                                       _large_subvectors, _pw_subvector_size,
            #                                                       name=prefixed_child_name)
            # _replace_child(parent, name, compressed_child, idx)
            return True

        elif isinstance(child, torch.nn.ConvTranspose2d):
            compressed_child = CompressedConvTranspose2d.from_uncompressed(
                child, _k, _kmeans_n_iters, _kmeans_fn, _large_subvectors, _pw_subvector_size, name=prefixed_child_name
            )
            _replace_child(parent, name, compressed_child, idx)
            return True

        elif isinstance(child, torch.nn.Linear):
            compressed_child = CompressedLinear.from_uncompressed(
                child, _k, _kmeans_n_iters, _kmeans_fn, _fc_subvector_size, name=prefixed_child_name
            )
            _replace_child(parent, name, compressed_child, idx)
            return True

        else:
            return False

    apply_recursively_to_model_special(_compress_and_replace_layer, model)
    multi_compression(model)
    return model
