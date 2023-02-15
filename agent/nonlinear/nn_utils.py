# Import modules
import torch
import torch.nn as nn
import numpy as np


def weights_init_(layer, init="kaiming", activation="relu"):
    """
    Initializes the weights for a fully connected layer of a neural network.
    Parameters
    ----------
    layer : torch.nn.Module
        The layer to initialize
    init : str
        The type of initialization to use, one of 'xavier_uniform',
        'xavier_normal', 'uniform', 'normal', 'orthogonal', 'kaiming_uniform',
        'default', by default 'kaiming_uniform'.
    activation : str
        The activation function in use, used to calculate the optimal gain
        value.
    """
    if "weight" in dir(layer):
        gain = torch.nn.init.calculate_gain(activation)

        if init == "xavier_uniform":
            torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
        elif init == "xavier_normal":
            torch.nn.init.xavier_normal_(layer.weight, gain=gain)
        elif init == "uniform":
            torch.nn.init.uniform_(layer.weight) / layer.in_features
        elif init == "normal":
            torch.nn.init.normal_(layer.weight) / layer.in_features
        elif init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight)
        elif init == "zeros":
            torch.nn.init.zeros_(layer.weight)
        elif init == "kaiming_uniform" or init == "default" or init is None:
            # PyTorch default
            return
        else:
            raise NotImplementedError(f"init {init} not implemented yet")

    if "bias" in dir(layer):
        torch.nn.init.constant_(layer.bias, 0)


def soft_update(target, source, tau):
    """
    Updates the parameters of the target network towards the parameters of
    the source network by a weight average depending on tau. The new
    parameters for the target network are:
        ((1 - τ) * target_parameters) + (τ * source_parameters)
    Parameters
    ----------
    target : torch.nn.Module
        The target network
    source : torch.nn.Module
        The source network
    tau : float
        The weighting for the weighted average
    """
    with torch.no_grad():
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            # Use in-place operations mul_ and add_ to avoid
            # copying tensor data
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)


def hard_update(target, source):
    """
    Sets the parameters of the target network to the parameters of the
    source network. Equivalent to soft_update(target,  source, 1)
    Parameters
    ----------
    target : torch.nn.Module
        The target network
    source : torch.nn.Module
        The source network
    """
    with torch.no_grad():
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(param.data)


def init_layers(layers, init_scheme):
    """
    Initializes the weights for the layers of a neural network.
    Parameters
    ----------
    layers : list of nn.Module
        The list of layers
    init_scheme : str
        The type of initialization to use, one of 'xavier_uniform',
        'xavier_normal', 'uniform', 'normal', 'orthogonal', by default None.
        If None, leaves the default PyTorch initialization.
    """
    def fill_weights(layers, init_fn):
        for i in range(len(layers)):
            init_fn(layers[i].weight)

    if init_scheme.lower() == "xavier_uniform":
        fill_weights(layers, nn.init.xavier_uniform_)
    elif init_scheme.lower() == "xavier_normal":
        fill_weights(layers, nn.init.xavier_normal_)
    elif init_scheme.lower() == "uniform":
        fill_weights(layers, nn.init.uniform_)
    elif init_scheme.lower() == "normal":
        fill_weights(layers, nn.init.normal_)
    elif init_scheme.lower() == "orthogonal":
        fill_weights(layers, nn.init.orthogonal_)
    elif init_scheme is None:
        # Use PyTorch default
        return


def _calc_conv_outputs(in_height, in_width, kernel_size, dilation=1, padding=0,
                       stride=1):
    """
    Calculates the output height and width given in input height and width and
    the kernel size.
    Parameters
    ----------
    in_height : int
        The height of the input image
    in_width : int
        The width of the input image
    kernel_size : tuple[int, int] or int
        The kernel size
    dilation : tuple[int, int] or int
        Spacing between kernel elements, by default 1
    padding : tuple[int, int] or int
        Padding added to all four sides of the input, by default 0
    stride : tuple[int, int] or int
        Stride of the convolution, by default 1
    Returns
    -------
    tuple[int, int]
        The output width and height
    """
    # Reshape so that kernel_size, padding, dilation, and stride have one
    # element per dimension
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    if isinstance(padding, int):
        padding = [padding] * 2
    if isinstance(dilation, int):
        dilation = [dilation] * 2
    if isinstance(stride, int):
        stride = [stride] * 2

    out_height = in_height + 2 * padding[0] - dilation[0] * (
        kernel_size[0] - 1) - 1
    out_height //= stride[0]

    out_width = in_width + 2 * padding[1] - dilation[1] * (
        kernel_size[1] - 1) - 1
    out_width //= stride[1]

    return out_height + 1, out_width + 1


def _get_activation(activation):
    """
    Returns an activation operation given a string describing the activation
    operation
    Parameters
    ----------
    activation : str
        The string representation of the activation operation, one of 'relu',
        'tanh'
    Returns
    -------
    nn.Module
        The activation function
    """
    # Set the activation funcitons
    if activation.lower() == "relu":
        act = nn.ReLU()
    elif activation.lower() == "tanh":
        act = nn.Tanh()
    else:
        raise ValueError(f"unknown activation {activation}")

    return act


def _construct_conv_linear(input_dim, num_actions, channels, kernel_sizes,
                           hidden_sizes, init, activation, single_output):
    """
    Constructs a number of convolutional layers and a sequence of
    densely-connected layers which operate on the output of the convolutional
    layers, returning the convolutional sequence and densely-connected sequence
    separately.
    This function is particularly suited to produce Q functions or
    Softmax policies, but can also be used to construct other approximators
    such as Gaussian policies or V functions (where `num_actions == 1` would
    actually be the number of state values to output, which is always 1; in
    such a case one should set `single_output = True`).
    This function construct a neural net which looks like:
        input   -->   convolutional   -->    densely-connected   -->   Output
                        layers                    layers
    and returns the convolutional and densely-connected layers separately.
    Parameters
    ----------
    input_dim : tuple[int, int, int]
        Dimensionality of state features, which should be (channels,
        height, width)
    num_actions : int
        If `single_output` is `True`, then this should be the dimensionality of
        the action, since then the action will be concatenated with the input
        to the linear layers. If `single_output` is `False`, then this should
        be the number of discrete available actions in the environment, and the
        network will output `num_actions` action values.
    channels : array-like[int]
        The number of channels in each hidden convolutional layer
    kernel_sizes : array-like[int]
        The number of channels in each consecutive convolutional layer
    hidden_sizes : array-like[int]
        The number of units in each consecutive fully connected layer
    init : str
        The initialization scheme to use for the weights, one of
        'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
        'orthogonal', by default None. If None, leaves the default
        PyTorch initialization.
    activation : indexable[str] or str
        The activation function to use; each element should be one of
        'relu', 'tanh'
    single_output : bool
        Whether or not the network should have a single output. If `True`, then
        the action is concatenated with the input the the linear layers. If
        `False`, then `num_actions` are outputted.
    """
    # Ensure the number of channels == the number of kernels
    if len(channels) != len(kernel_sizes):
        kernels = len(kernel_sizes)
        channels = len(channels)
        raise ValueError("must have the same number of channels and " +
                         f"kernels but got {channels} channels " +
                         f"and {kernels} kernels")

    if isinstance(activation, str):
        act = [_get_activation(activation)] * (len(channels) +
                                               len(hidden_sizes))
    elif len(channels) != len(activations):
        activations = len(activations)
        channels = len(channels)
        raise ValueError("must have the same number of channels and " +
                         f"activations but got {channels} channels " +
                         f"and {activations} activations")

    # Convolutional layers
    conv = []  # List of sequential convolutional layers and activations
    in_channels = input_dim[0]
    out_channels = channels[0]
    kernel = kernel_sizes[0]
    channel_size = input_dim[1:]
    for i in range(1, len(channels)):
        # Append the convolutional layer and activation to the list of layers
        conv.append(nn.Conv2d(in_channels, out_channels, kernel))
        conv.append(act[i-1])

        # Calculate the next channel size to be used later for the number of
        # inputs to the dense layers
        channel_size = _calc_conv_outputs(channel_size[0],
                                          channel_size[1], kernel)

        # Update some running variables for the convolutional layer sizes for
        # the next convolutional layer
        in_channels = out_channels
        out_channels = channels[i]
        kernel = kernel_sizes[i]

    # Append the last convolutional layer to the list of layers
    conv.append(nn.Conv2d(in_channels, out_channels, kernel))
    conv.append(act[len(channels)-1])
    channel_size = _calc_conv_outputs(channel_size[0],
                                      channel_size[1], kernel)

    # Ensure that the final output size of the convolutional layers
    # is non-negative
    if np.any(np.array(channel_size) < 0):
        raise ValueError("convolutions produce shape with negative size")

    # Construct the chain of convolutions and activations
    conv = nn.Sequential(*conv)
    conv.apply(lambda module: weights_init_(module, init))

    # Get the final number of elements of the output of the
    # convolutional layers
    conv_out = out_channels * np.prod(channel_size)

    # Linear layers
    linear = []  # List of dense connections and activations
    in_units = conv_out + (num_actions if single_output else 0)
    for i in range(len(hidden_sizes)):
        # Add a dense layer and activation to the list of operations for the
        # fuylly connected layers
        linear.append(nn.Linear(in_units, hidden_sizes[i]))
        linear.append(act[len(channels) + i])

        # Update the number of inputs to the next layer
        in_units = hidden_sizes[i]

    # Add the final dense layer
    if single_output:
        linear.append(nn.Linear(in_units, 1))
    else:
        linear.append(nn.Linear(in_units, num_actions))

    # Construct the chain of dense connections and activations
    linear = nn.Sequential(*linear)
    linear.apply(lambda module: weights_init_(module, init))

    return conv, linear
