"""
Determine spatial relationships between layers to relate their coordinates.
Coordinates are mapped from input-to-output (forward), but can
be mapped output-to-input (backward) by the inverse mapping too.
This helps crop and align feature maps among other uses.
"""

from __future__ import division
import numpy as np
from caffe import layers as L

PASS_THROUGH_LAYERS = ['AbsVal', 'BatchNorm', 'Bias', 'BNLL', 'Dropout',
                       'Eltwise', 'ELU', 'Log', 'LRN', 'Exp', 'MVN', 'Power',
                       'ReLU', 'PReLU', 'Scale', 'Sigmoid', 'Split', 'TanH',
                       'Threshold', 'Concat']


def conv_params(fn):
    """
    Extract the spatial parameters that determine the coordinate mapping:
    kernel size, stride, padding, and dilation.

    Implementation detail: Convolution, Deconvolution, and Im2col layers
    define these in the convolution_param message, while Pooling has its
    own fields in pooling_param. This method deals with these details to
    extract canonical parameters.
    """
    params = fn.params.get('convolution_param', fn.params)
    axis = params.get('axis', 1)
    ks = np.array(params['kernel_size'], ndmin=1)
    dilation = np.array(params.get('dilation', 1), ndmin=1)
    assert len({'pad_h', 'pad_w', 'kernel_h', 'kernel_w', 'stride_h',
                'stride_w'} & set(fn.params)) == 0, \
        'cropping does not support legacy _h/_w params'
    return (axis, np.array(params.get('stride', 1), ndmin=1),
            (ks - 1) * dilation + 1,
            np.array(params.get('pad', 0), ndmin=1))


def crop_params(fn):
    """
    Extract the crop layer parameters with defaults.
    """
    params = fn.params.get('crop_param', fn.params)
    axis = params.get('axis', 2)  # default to spatial crop for N, C, H, W
    offset = np.array(params.get('offset', 0), ndmin=1)
    return (axis, offset)


class UndefinedMapException(Exception):
    """
    Exception raised for layers that do not have a defined coordinate mapping.
    """
    pass


def coord_map(fn):
    """
    Define the coordinate mapping by its
    - axis
    - scale: output coord[i * scale] <- input_coord[i]
    - shift: output coord[i] <- output_coord[i + shift]
    s.t. the identity mapping, as for pointwise layers like ReLu, is defined by
    (None, 1, 0) since it is independent of axis and does not transform coords.
    """
    if fn.type_name in ['Convolution', 'Pooling', 'Im2col']:
        axis, stride, ks, pad = conv_params(fn)
        return axis, 1 / stride, (pad - (ks - 1) / 2) / stride
    elif fn.type_name == 'Deconvolution':
        axis, stride, ks, pad = conv_params(fn)
        return axis, stride, (ks - 1) / 2 - pad
    elif fn.type_name in PASS_THROUGH_LAYERS:
        return None, 1, 0
    elif fn.type_name == 'Crop':
        axis, offset = crop_params(fn)
        axis -= 1  # -1 for last non-coordinate dim.
        return axis, 1, - offset
    else:
        print("Encountered an unfamiliar layer: " + fn.type_name)
        raise UndefinedMapException


class AxisMismatchException(Exception):
    """
    Exception raised for mappings with incompatible axes.
    """
    pass


def compose(base_map, next_map):
    """
    Compose a base coord map with scale a1, shift b1 with a further coord map
    with scale a2, shift b2. The scales multiply and the further shift, b2,
    is scaled by base coord scale a1.
    """
    print "compose"
    ax1, a1, b1 = base_map
    ax2, a2, b2 = next_map
    print ax1, a1, b1, ax2, a2, b2
    if ax1 is None:
        ax = ax2
    elif ax2 is None or ax1 == ax2:
        ax = ax1
    else:
        print "AxisMismatchException"
        raise AxisMismatchException
    return ax, a1 * a2, a1 * b2 + b1


def inverse(coord_map):
    """
    Invert a coord map by de-scaling and un-shifting;
    this gives the backward mapping for the gradient.
    """
    ax, a, b = coord_map
    return ax, 1 / a, -b / a


def coord_map_from_to(top_from, top_to):
    """
    Determine the coordinate mapping betweeen a top (from) and a top (to).
    Walk the graph to find a common ancestor while composing the coord maps for
    from and to until they meet. As a last step the from map is inverted.
    """
    # We need to find a common ancestor of top_from and top_to.
    # We'll assume that all ancestors are equivalent here (otherwise the graph
    # is an inconsistent state (which we could improve this to check for)).
    # For now use a brute-force algorithm.

    def collect_bottoms(top):
        """
        Collect the bottoms to walk for the coordinate mapping.
        The general rule is that all the bottoms of a layer can be mapped, as
        most layers have the same coordinate mapping for each bottom.
        Crop layer is a notable exception. Only the first/cropped bottom is
        mappable; the second/dimensions bottom is excluded from the walk.
        """
        bottoms = top.fn.inputs
        if top.fn.type_name == 'Crop':
            bottoms = bottoms[:1]
        if top.fn.type_name == 'Concat':
            print "Concat layer bottoms:"
            print (top.fn.inputs[0], top.fn.inputs[0].fn.type_name, top.fn.inputs[0].fn.params)
            print (top.fn.inputs[1], top.fn.inputs[1].fn.type_name, top.fn.inputs[1].fn.params)
        return bottoms

    # walk back from top_from, keeping the coord map as we go
    from_maps = {top_from: (None, 1, 0)}
    frontier = {top_from}
    while frontier:
        top = frontier.pop()
        try:
            bottoms = collect_bottoms(top)
            print(11)
            for bottom in bottoms:
                print(12)
                from_maps[bottom] = compose(from_maps[top], coord_map(top.fn))
                print(13)
                frontier.add(bottom)
                print(14)
                print "Top-bottom pair:"
                print (top, top.fn.type_name, top.fn.params)
                print (bottom, bottom.fn.type_name, bottom.fn.params)
        except UndefinedMapException:
            print "Exception reached"
            pass
    print len(from_maps)
    # for k, v in from_maps.items():
    #     print k.fn.params
    print [(k.fn.type_name, k.fn.params) for k, v in from_maps.items() if len(k.fn.params) > 0]
    #print frontier

    # now walk back from top_to until we hit a common blob
    to_maps = {top_to: (None, 1, 0)}
    frontier = {top_to}
    while frontier:
        top = frontier.pop()
        if top in from_maps:
            print "common ancestor: "
            print top
            print top.fn.type_name
            print top.fn.params
            print top.fn.ntop
            return compose(to_maps[top], inverse(from_maps[top]))
        try:
            bottoms = collect_bottoms(top)
            for bottom in bottoms:
                to_maps[bottom] = compose(to_maps[top], coord_map(top.fn))
                frontier.add(bottom)
        except UndefinedMapException:
            continue
    print to_maps
    #print frontier

    # if we got here, we did not find a blob in common
    raise RuntimeError('Could not compute map between tops; are they '
                       'connected by spatial layers?')


def crop(top_from, top_to):
    """
    Define a Crop layer to crop a top (from) to another top (to) by
    determining the coordinate mapping between the two and net spec'ing
    the axis and shift parameters of the crop.
    """
    print "New crop:"
    print top_from.fn.type_name
    # print top_from.fn.inputs
    # print top_from.fn.params
    # print top_from.fn.tops
    print top_to.fn.type_name
    ax, a, b = coord_map_from_to(top_from, top_to)
    assert (a == 1).all(), 'scale mismatch on crop (a = {})'.format(a)
    assert (b <= 0).all(), 'cannot crop negative offset (b = {})'.format(b)
    assert (np.round(b) == b).all(), 'cannot crop noninteger offset ' \
        '(b = {})'.format(b)
    return L.Crop(top_from, top_to,
                  crop_param=dict(axis=ax + 1,  # +1 for first cropping dim.
                                  offset=list(-np.round(b).astype(int))))
