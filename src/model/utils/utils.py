import tensorlayer as tl
import tensorflow as tf
import json


def require_a_kwarg(name, kwargs):
    var = None
    for k, v in kwargs.items():
        if k is name:
            var = v
    if not var:
        raise Exception(("Missing a parameter '%s', call the method with %s=XXX" % (name, name)))
    else:
        return var


def flatten_and_concat_tensors(name_prefix, tensor_dict):
    flattened_input_list = []
    for name, tensor in tensor_dict.items():
        tensor_shape = tensor.get_shape().as_list()
        new_shape = [-1] + [tensor_shape[i] for i in range(2, len(tensor_shape))]

        input_layer = tl.layers.InputLayer(inputs=tensor,
                                           name=name_prefix + 'INPUT_LAYER_' + name)
        reshape_layer = tl.layers.ReshapeLayer(layer=input_layer,
                                               shape=new_shape,
                                               name=name_prefix + 'RESHAPE_LAYER_' + name)

        flatten_layer = tl.layers.FlattenLayer(layer=reshape_layer,
                                               name=name_prefix + 'FLATTEN_LAYER_' + name)
        flattened_input_list.append(flatten_layer)
    flattened_input_list = tl.layers.ConcatLayer(layer=flattened_input_list,
                                                 concat_dim=1,
                                                 name=name_prefix + 'CONCAT_LOW_DIM_INPUT_LAYER')
    return flattened_input_list


def load_json(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
        return res

# def merge_two_dict(*args):
#     z = a.copy()
#     z.update(b)
#     return z


if __name__ == '__main__':
    a = {'a': 1}
    b = {'a': 2}
    # print(merge_two_dict(a, b))
