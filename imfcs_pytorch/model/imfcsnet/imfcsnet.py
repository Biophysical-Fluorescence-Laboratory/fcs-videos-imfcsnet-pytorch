"""PyTorch port of the original ImFCSNet from the original paper.

The original network was designed in Tensorflow 1.0, before being ported over to Tensorflow 2.0 to allow for ONNX-related exporting. This led to a **lot** of technical debt that was never really cleaned up.

To facilitate easier development, we opted to reimplement the network in PyTorch to facilitate easier development, and to also allow for easier contributions (as PyTorch is almost the de-facto standard for research-related machine learning - https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/).

The networks are split up into different blocks, but the functionality should be identical to what is expected from the original network code (https://github.com/Biophysical-Fluorescence-Laboratory/ImFCS_FCSNet_ImFCSNet/blob/master/imfcsnet/nn_model/imfcsnet.py).

To confirm that our implementation is correct, we ran `model.summary()` on the original codebase, and compared the parameter counts to our implementation through `torchinfo.summary()`. More in-depth reproducibility studies were also done and documented under the issues tab of the working Github repository.
"""

import torch
from torch import nn

# Typing-specific imports.
from typing import Tuple


class Conv1DBatchNormBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        norm_momentum: float = 0.1,  # See notes, PyTorch has a different momentum definition.
        activation: nn.Module = nn.ReLU,
        norm_layer: nn.Module = None,
    ):
        """Reimplementation of the _CONV1D_BN_RELU block from the original code. This should be the fundamental building block of all stride-1 Conv1D layers.

        Note:
        - In PyTorch, the in_channels needs to be specified, unlike Tensorflow where it can be infered during execution. This means the original `filters` arg maps to the `out_channels` arg.
        - The BatchNorm momentum convention in PyTorch is different from Tensorflow, see https://stackoverflow.com/questions/48345857/batchnorm-momentum-convention-pytorch
        - Following the design of ResNets in PyTorch, the shortcut skip connection is embedded into the block, rather than being a separate entity.

        Args:
            in_channels (int): The number of channels of the input.
            out_channels (int): The number of channels to use for the output.
            filter_size (int): Filter kernel size.
            norm_momentum (float, optional): Batch normalization momentum, note the difference in PyTorch's batch norm vs Tensorflow. Defaults to 0.1.
            norm_layer (nn.Module, optional): Override the default batch normalization layer. Defaults to None.
        """
        super(Conv1DBatchNormBlock, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=filter_size,
        )

        if norm_layer is not None:
            # Potentially allow the swapping out of batchnorm layers.
            self.norm = norm_layer
        else:
            # Otherwise, default to the standard 1D BatchNorm layer of the original
            self.norm = nn.BatchNorm1d(
                num_features=out_channels, momentum=norm_momentum, eps=0.001
            )

        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class SpatialAggregationBlock(nn.Module):
    def __init__(
        self,
        spatial_in_channels: int,
        spatial_out_channels: int,
        spatial_filter_kernel_size: Tuple[int],
        norm_momentum: float = 0.1,
        activation: nn.Module = nn.ReLU,
        norm_layer: nn.Module = None,
    ):
        """This is a grouping of all the functions involved for the spatial aggregation aspect of ImFCSNet. This converts the 3x3 inputs into a 1x1 temporal-only signal which can be processed by the rest of the network.

        This is primarily done for neatness and separation of concerns.

        Args:
            spatial_in_channels (int): The number of channels of the input.
            spatial_out_channels (int): The number of channels to use for the output.
            spatial_filter_kernel_size (Tuple[int]): Filter kernel size, of shape (FRAMES, WIDTH, HEIGHT).
            norm_momentum (float, optional): Batch normalization momentum, note the difference in PyTorch's batch norm vs Tensorflow. Defaults to 0.1.
            activation (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
            norm_layer (nn.Module, optional): Override the default batch normalization layer. Defaults to None.
        """
        super(SpatialAggregationBlock, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels=spatial_in_channels,
            out_channels=spatial_out_channels,
            kernel_size=spatial_filter_kernel_size,
        )

        if norm_layer is not None:
            # Potentially allow the swapping out of batchnorm layers.
            self.norm = norm_layer
        else:
            # Otherwise, default to the standard 1D BatchNorm layer of the original
            self.norm = nn.BatchNorm1d(
                num_features=spatial_out_channels, momentum=norm_momentum, eps=0.001
            )

        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We remove the extraneous 1x1 dimension to cast this into a true 1D temporal signal.
        # This replaces the Lambda layers of the original model.
        x = self.conv3d(x).squeeze((-1, -2))
        x = self.activation(self.norm(x))

        return x


class StridedConv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        filter_stride: int,
        norm_momentum: float = 0.1,
        activation: nn.Module = nn.ReLU,
        norm_layer: nn.Module = None,
    ):
        """This is a wrapped version of the `_Conv1D_stride` layer from the original model. This seems to be primarily for purposes of dimensionality reduction, and happens before the non-strided Conv1D blocks are applied.

        Args:
            in_channels (int): The number of channels of the input.
            out_channels (int): The number of channels to use for the output.
            kernel_size (int): Filter kernel size, of shape (FRAMES).
            filter_stride (int): Stride to use for the temporal convolutional filter.
            norm_momentum (float, optional): Batch normalization momentum, note the difference in PyTorch's batch norm vs Tensorflow. Defaults to 0.1.
            activation (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
            norm_layer (nn.Module, optional): Override the default batch normalization layer. Defaults to None.
        """
        super(StridedConv1DBlock, self).__init__()

        self.conv1d_stride = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=filter_stride,
        )

        if norm_layer is not None:
            # Potentially allow the swapping out of batchnorm layers.
            self.norm = norm_layer
        else:
            # Otherwise, default to the standard 1D BatchNorm layer of the original
            self.norm = nn.BatchNorm1d(
                num_features=out_channels, momentum=norm_momentum, eps=0.001
            )

        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d_stride(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Conv1dGroup(nn.Module):
    def __init__(
        self,
        channels: int,
        num_stages: int,
        blocks_per_stage: int,
        filter_size: int,
        norm_momentum: float = 0.1,
        activation: nn.Module = nn.ReLU,
        norm_layer: nn.Module = None,
    ):
        """This is a grouping of the Conv1D blocks, which handle the temporal aggregation aspect.

        Notes:
        - To handle the dynamic building of layers, we follow the convention from the ConvNeXt repo (https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py), where we can access layers individually as necessary. This is needed because the shortcut connections happen every 2-blocks.
        - As described in the docstring of Conv1DBatchNormBlock, the shortcut connections are baked into the blocks.

        Args:
            channels (int): Number of channels to use. We follow the implementation of the original paper, where the channel counts remain constant throughout every single block.
            num_stages (int): How many stages to use. A skip connection is applied over each stage.
            blocks_per_stage (int): Number of blocks to use per-stage.
            filter_size (int): Filter kernel size, of shape (FRAMES).
            norm_momentum (float, optional): Batch normalization momentum, note the difference in PyTorch's batch norm vs Tensorflow. Defaults to 0.1.
            activation (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
            norm_layer (nn.Module, optional): Override the default batch normalization layer. Defaults to None.
        """
        super(Conv1dGroup, self).__init__()

        # Run a sanity check to ensure that the number of layers is directly divisible by the layers_per_residual.
        # assert (
        #     num_layers_total % layers_per_residual == 0
        # ), f"num_layers is not directly divisible by layers_per_residual ({num_layers_total} % {layers_per_residual} = {num_layers_total % layers_per_residual}). This means there are leftover layers which might not have access to a shortcut connection."

        # self.num_of_stages = num_layers_total // layers_per_residual
        self.num_stages = num_stages

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[
                    Conv1DBatchNormBlock(
                        in_channels=channels,
                        out_channels=channels,
                        filter_size=filter_size,
                        norm_momentum=norm_momentum,
                        activation=activation,
                        norm_layer=norm_layer,
                    )
                    for j in range(blocks_per_stage)
                ]
            )
            self.stages.append(stage)

        # We need to calculate the truncation amount due to the lack of padding.
        # This means each stage will cause a corresponding reduction in the temporal dimension of our signal.
        # Since the skip connection requires an addition, this means we need to slice accordingly.
        # TODO: Do we want to follow the original method of slicing off the tail only?
        self.truncate_amount = 2 * filter_size - 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_stages):
            # Slice the tail of the feature dimension
            identity = x[:, :, : -self.truncate_amount]
            x = self.stages[i](x) + identity

        return x


class DenseConv1x1Group(nn.Module):
    def __init__(
        self,
        channels: int,
        num_stages: int,
        blocks_per_stage: int,
        norm_momentum: float = 0.1,
        activation: nn.Module = nn.ReLU,
        norm_layer: nn.Module = None,
    ):
        """This is the grouping of 1x1 conv blocks that was used in the original network. The renaming to 'DenseConv1x1' is because the kernel size of 1 used in these layers is fundamentally the same as linear layers, which means these can simply be reformulated as FC layers. This renaming might avoid confusion and provide a clearer separation of concerns.

        Notes:
        - A similar reformulation was done by the timm library (https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py#L82) for converting Conv1x1 to Linear layers.
        - Tentatively, we do not reformulate to Linear layers.
        - This is very similar to the Conv1dGroup class, just without the truncations (kernel size 1 means no padding is needed)

        Args:
            channels (int): Number of channels to use. We follow the implementation of the original paper, where the channel counts remain constant throughout every single block.
            num_stages (int): How many stages to use. A skip connection is applied over each stage.
            blocks_per_stage (int): Number of blocks to use per-stage.
            norm_momentum (float, optional): Batch normalization momentum, note the difference in PyTorch's batch norm vs Tensorflow. Defaults to 0.1.
            activation (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
            norm_layer (nn.Module, optional): Override the default batch normalization layer. Defaults to None.
        """
        super(DenseConv1x1Group, self).__init__()

        self.num_stages = num_stages

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = nn.Sequential(
                *[
                    Conv1DBatchNormBlock(
                        in_channels=channels,
                        out_channels=channels,
                        filter_size=1,  # With a filter size of 1, this is fundamentally just a linear layer.
                        norm_momentum=norm_momentum,
                        activation=activation,
                        norm_layer=norm_layer,
                    )
                    for j in range(blocks_per_stage)
                ]
            )
            self.stages.append(stage)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_stages):
            # Slice the tail of the feature dimension
            identity = x
            x = self.stages[i](x) + identity

        return x


class ImFCSNet(nn.Module):
    def __init__(
        self,
        spatial_layer: nn.Module,
        strided_conv_layer: nn.Module,
        conv_group: nn.Module,
        dense_mixing_group: nn.Module,
        fc_in_features: int,
        fc_out_features: int,
        use_original_init: bool = False,
    ):
        """Wrapper around all of the individual building blocks.

        Reimplementation of the model produced by the `imfcsnet.nn_model.gen_model` function. Following PyTorch semantics means that there are a few items which are grouped differently, but the fundamental effect should be identical.

        Args:
            spatial_layer (nn.Module): Spatial aggregation layer.
            strided_conv_layer (nn.Module): Initial strided temporal convolution layer.
            conv_group (nn.Module): The collection of temporal convolution blocks.
            dense_mixing_group (nn.Module): The collection of 1x1 convolutions to mix temporal features.
            fc_in_features (int): The number of input features to the output regression/classification head.
            fc_out_features (int): Number of output neurons. Generally reflects the number of targets.
            use_original_init (bool, optional): Whether to replicate the original model's initialization scheme, which differs from PyTorch's default preference of Kaiming Uniform. Defaults to False.
        """
        super(ImFCSNet, self).__init__()

        # Spatial layers. These handle the initial 3D 3x3 inputs and cast them into a 1D-compatible form.
        self.spatial_layer = spatial_layer
        self.strided_conv_layer = strided_conv_layer
        self.conv_group = conv_group
        self.dense_mixing_group = dense_mixing_group

        # This MLP layer converts the feature-space into the output nodes that are required.
        # In the case of the original D-regression network, this would be 1
        self.fc = nn.Linear(fc_in_features, fc_out_features)

        if use_original_init:
            self.source_paper_weight_init()

    def source_paper_weight_init(self):
        """This function implements the functionality to reproduce the original paper's intialization scheme.

        Since the original paper was implemented in Tensorflow, there are a few defaults which differ from PyTorch. For example, the linear layers default to glorot_uniform (as opposed to kaiming_uniform in PyTorch), and the original codebase used he_normal for all Conv layers (as opposed to kaiming_uniform in PyTorch).

        Changing initializations theoretically should not make that big of a difference, but for the purposes of reproducibility, this function attempts to reproduce things as well as possible.

        Note that there are other aspects which differ between PyTorch and Tensorflow. For example, optimizer epsilon terms etc.
        """

        def init_weights(m):
            # Linear/Dense layers in Tensorflow default to glorot_uniform
            if isinstance(m, nn.Linear):
                print(f"Initializing {m} with Xavier uniform.")
                nn.init.xavier_uniform_(m.weight)
            # Conv layers in the original paper followed the config file, which used he_normal
            # PyTorch uses he_uniform
            if isinstance(m, (nn.Conv3d, nn.Conv1d)):
                print(f"Initializing {m} with Kaiming normal.")
                nn.init.kaiming_normal_(m.weight)

        super().apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_layer(x)
        x = self.strided_conv_layer(x)
        x = self.conv_group(x)
        x = self.dense_mixing_group(x)

        # Temporal averaging as per the original model.
        x = torch.mean(x, dim=-1)

        x = self.fc(x)
        return x


def build_imfcsnet(
    filter_channels: int,
    spatial_agg_block_kernel_size: Tuple[int],
    strided_conv_layer_kernel_size: int,
    strided_conv_layer_filter_stride: int,
    conv1d_group_stages: int,
    conv1d_group_blocks_per_stage: int,
    conv1d_group_filter_size: int,
    dense_mixing_num_stages: int,
    dense_mixing_blocks_per_stage: int,
    output_neurons: int,
    use_original_init: bool = False,
) -> ImFCSNet:
    """Function to build ImFCSNet by initializing the individual building blocks by config parameters.

    Args:
        filter_channels (int): The filter channel counts to use across the network. Following the original paper, the channel counts do not change throughout the entire network.
        spatial_agg_block_kernel_size (Tuple[int]): Convolutional kernel size of the spatial aggregation block. Original paper uses (200, 3, 3)
        strided_conv_layer_kernel_size (int): Convolutional kernel size of the strided temporal aggregation block.
        strided_conv_layer_filter_stride (int): Stride of the convolutional filter in the strided temporal aggregation block.
        conv1d_group_stages (int): Number of stages in the group of Conv1d blocks.
        conv1d_group_blocks_per_stage (int): Number of blocks per stage in the group of Conv1d blocks.
        conv1d_group_filter_size (int): Convolutional kernel size of the group of Conv1d blocks.
        dense_mixing_num_stages (int): Number of stages to use in the dense mixing block with size-1 convolutions.
        dense_mixing_blocks_per_stage (int): Number of blocks per stage in the dense mixing block with size-1 convolutions.
        output_neurons (int): Number of output neurons. Generally reflects the number of targets.
        use_original_init (bool, optional): Whether to replicate the original model's initialization scheme, which differs from PyTorch's default preference of Kaiming Uniform. Defaults to False.

    Returns:
        ImFCSNet: Constructed ImFCSNet.
    """
    spatial_agg_block = SpatialAggregationBlock(
        spatial_in_channels=1,
        spatial_out_channels=filter_channels,
        spatial_filter_kernel_size=spatial_agg_block_kernel_size,  # (FILTER_3D_SIZE, SPATIAL_FILTER_SIZE, SPATIAL_FILTER_SIZE)
    )
    strided_conv_layer = StridedConv1DBlock(
        in_channels=filter_channels,
        out_channels=filter_channels,
        kernel_size=strided_conv_layer_kernel_size,  # FILTER_1_SIZE
        filter_stride=strided_conv_layer_filter_stride,
    )
    conv1d_group = Conv1dGroup(
        channels=filter_channels,
        num_stages=conv1d_group_stages,
        blocks_per_stage=conv1d_group_blocks_per_stage,
        filter_size=conv1d_group_filter_size,  # TEMPORAL_FILTER_SIZE
    )
    dense_mixing = DenseConv1x1Group(
        channels=filter_channels,
        num_stages=dense_mixing_num_stages,
        blocks_per_stage=dense_mixing_blocks_per_stage,
    )

    # Wrapped as a whole model.
    imfcsnet = ImFCSNet(
        spatial_layer=spatial_agg_block,
        strided_conv_layer=strided_conv_layer,
        conv_group=conv1d_group,
        dense_mixing_group=dense_mixing,
        fc_in_features=filter_channels,
        fc_out_features=output_neurons,
        use_original_init=use_original_init,
    )

    return imfcsnet


if __name__ == "__main__":
    # This runs a basic test that attempts to compile the model to check if it works.
    # The values shown here are based on the original config file for ImFCSNet provided in the official Github page (https://github.com/ImagingFCS/ImFCS_FCSNet_ImFCSNet/blob/master/imfcsnet/Configurations/CNN.py)

    # Config variables.
    FILTER_CHANNELS = 45

    # Generate the input. By default, this should be of shape (1, 2500, 3, 3) -> (C, T, H, W). Add the batch dimension.
    input_arr = torch.randn(1, 2500, 3, 3).unsqueeze(0)
    print(f"Input: {input_arr.size()}")

    print("Testing spatial aggregation block.")
    spatial_agg_block = SpatialAggregationBlock(
        spatial_in_channels=1,
        spatial_out_channels=FILTER_CHANNELS,
        spatial_filter_kernel_size=(
            200,
            3,
            3,
        ),  # (FILTER_3D_SIZE, SPATIAL_FILTER_SIZE, SPATIAL_FILTER_SIZE)
    )
    x = spatial_agg_block(input_arr)
    print(f"Post-spatial aggregation block: {x.size()}")

    print("Testing StridedConv1d layer.")
    strided_conv_layer = StridedConv1DBlock(
        in_channels=FILTER_CHANNELS,
        out_channels=FILTER_CHANNELS,
        kernel_size=100,  # FILTER_1_SIZE
        filter_stride=8,
    )
    x = strided_conv_layer(x)
    print(f"Post-StridedConv1DBlock: {x.size()}")

    print("Testing Conv1D Group.")
    conv1d_group = Conv1dGroup(
        channels=FILTER_CHANNELS,
        num_stages=2,
        blocks_per_stage=2,
        filter_size=50,  # TEMPORAL_FILTER_SIZE
    )
    x = conv1d_group(x)
    print(f"Post-Conv1dGroup: {x.size()}")

    print("Testing Conv1x1 Group, renamed as DenseConv1x1Group.")
    dense_mixing = DenseConv1x1Group(
        channels=FILTER_CHANNELS,
        num_stages=6,
        blocks_per_stage=2,
    )
    x = dense_mixing(x)
    print(f"Post-DenseConv1x1Group: {x.size()}")
    print("-" * 20)

    # Wrapped as a whole model.
    print("Testing a whole ImFCSNet.")
    imfcsnet = ImFCSNet(
        spatial_layer=spatial_agg_block,
        strided_conv_layer=strided_conv_layer,
        conv_group=conv1d_group,
        dense_mixing_group=dense_mixing,
        fc_in_features=FILTER_CHANNELS,
        fc_out_features=1,
    )
    print(f"Input to ImFCSNet: {input_arr.size()}")
    print(f"Output of ImFCSNet: {imfcsnet(input_arr)}: ({imfcsnet(input_arr).size()})")

    print("Finally, testing the generalizability to variable input lengths.")
    input_arr = torch.randn(1, 50000, 3, 3).unsqueeze(0)
    print(f"Input to ImFCSNet: {input_arr.size()}")
    print(f"Output of ImFCSNet: {imfcsnet(input_arr)}: ({imfcsnet(input_arr).size()})")

    from torchinfo import summary

    batch_size = 4
    summary(imfcsnet, input_size=(batch_size, 1, 2500, 3, 3))
