"""
Exports a pytorch model to an ONNX format, and then converts from the
ONNX to a Tensorflow serving protobuf file.
Running example:
python3 pytorch_to_tf_serving.py \
 --onnx-file text.onnx \
 --meta-file text.meta \
 --export-dir serving_model/
"""

import logging
import argparse

import tensorflow as tf
from tensorflow.python.saved_model import utils as smutils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from onnx_tf.backend import prepare
import onnx
import torch

from text_class_learner import TextClassificationLearner

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def export_onnx(model, dummy_input, file, input_names, output_names, num_inputs):
    """
    Converts a Pytorch model to the ONNX format and saves the .onnx model file.
    The first dimension of the input nodes are of size N, where N is the
    minibatch size. This dimensions is here replaced by an arbitrary string
    which the ONNX -> TF library interprets as the '?' dimension in Tensorflow.
    This process is applied because the input minibatch size should be of an
    arbitrary size.
    :param model: Pytorch model instance with loaded weights
    :param dummy_input: tuple, dummy input numpy arrays that the model
        accepts in the inference time. E.g. for the Text+Image model, the
        tuple would be (np.float32 array of N x W x H x 3, np.int64 array of
        N x VocabDim). Actual numpy arrays values don't matter, only the shape
        and the type must match the model input shape and type. N represents
        the minibatch size and can be any positive integer. True batch size
        is later handled when exporting the model from the ONNX to TF format.
    :param file: string, Path to the exported .onnx model file
    :param input_names: list of strings, Names assigned to the input nodes
    :param output_names: list of strings, Names assigned to the output nodes
    :param num_inputs: int, Number of model inputs (e.g. 2 for Text and Image)
    """
    # List of onnx.export function arguments:
    # https://github.com/pytorch/pytorch/blob/master/torch/onnx/utils.py
    # ISSUE: https://github.com/pytorch/pytorch/issues/14698
    torch.onnx.export(
        model,
        args=dummy_input,
        input_names=input_names,
        output_names=output_names,
        f=file,
    )

    # Reload model to fix the batch size
    model = onnx.load(file)
    model = make_variable_batch_size(num_inputs, model)
    onnx.save(model, file)

    log.info("Exported ONNX model to {}".format(file))


def make_variable_batch_size(num_inputs, onnx_model):
    """
    Changes the input batch dimension to a string, which makes it variable.
    Tensorflow interpretes this as the "?" shape.
    `num_inputs` must be specified because `onnx_model.graph.input` is a list
    of inputs of all layers and not just model inputs.
    :param num_inputs: int, Number of model inputs (e.g. 2 for Text and Image)
    :param onnx_model: ONNX model instance
    :return: ONNX model instance with variable input batch size
    """
    for i in range(num_inputs):
        onnx_model.graph.input[i].type.tensor_type.shape.dim[0].dim_param = "batch_size"
    return onnx_model


def export_tf_proto(onnx_file, meta_file):
    """
    Exports the ONNX model to a Tensorflow Proto file.
    The exported file will have a .meta extension.
    :param onnx_file: string, Path to the .onnx model file
    :param meta_file: string, Path to the exported Tensorflow .meta file
    :return: tuple, input and output tensor dictionaries. Dictionaries have a
        {tensor_name: TF_Tensor_op} structure.
    """
    model = onnx.load(onnx_file)

    # Convert the ONNX model to a Tensorflow graph
    tf_rep = prepare(model)
    output_keys = tf_rep.outputs
    input_keys = tf_rep.inputs

    tf_dict = tf_rep.tensor_dict
    input_tensor_names = {key: tf_dict[key] for key in input_keys}
    output_tensor_names = {key: tf_dict[key] for key in output_keys}

    tf_rep.export_graph(meta_file)
    log.info("Exported Tensorflow proto file to {}".format(meta_file))
    return input_tensor_names, output_tensor_names


def export_for_serving(meta_path, export_dir, input_tensors, output_tensors):
    """
    Exports the Tensorflow .meta model to a frozen .pb Tensorflow serving
       format.
    :param meta_path: string, Path to the .meta TF proto file.
    :param export_dir: string, Path to directory where the serving model will
        be exported.
    :param input_tensor: dict, Input tensors dictionary of
        {name: TF placeholder} structure.
    :param output_tensors: dict, Output tensors dictionary of {name: TF tensor}
        structure.
    """
    g = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_def = tf.GraphDef()

    with g.as_default():
        with open(meta_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        # name argument must explicitly be set to an empty string, otherwise
        # TF will prepend an `import` scope name on all operations
        tf.import_graph_def(graph_def, name="")

        tensor_info_inputs = {
            name: smutils.build_tensor_info(in_tensor)
            for name, in_tensor in input_tensors.items()
        }

        tensor_info_outputs = {
            name: smutils.build_tensor_info(out_tensor)
            for name, out_tensor in output_tensors.items()
        }

        prediction_signature = signature_def_utils.build_signature_def(
            inputs=tensor_info_inputs,
            outputs=tensor_info_outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME,
        )

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            signature_def_map={"predict_images": prediction_signature},
        )
        builder.save()

        log.info("Input info:\n{}".format(tensor_info_inputs))
        log.info("Output info:\n{}".format(tensor_info_outputs))


def main(args):

    # Initialize pytorch model
    learner = TextClassificationLearner.load(args.pretrained_path)

    model = learner.model

    n_docs = 5
    n_sents = 7
    sents_len = 100
    sents = torch.ones(
        n_docs, n_sents, sents_len, dtype=torch.long
    )  # TODO: variable seq len?

    dummy_inputs = sents
    input_names = "sents"
    # img_input = torch.randn(1, 3, 224, 224)

    # input_names = ['input_img']
    # output_names = ['confidences']

    # Use a tuple if there are multiple model inputs
    # dummy_inputs = (img_input)
    output_names = ("activations", "word_attn_weight", "sent_attn_weight", "rec_loss")

    export_onnx(
        model,
        dummy_inputs,
        args.onnx_file,
        input_names=input_names,
        output_names=output_names,
        num_inputs=1,
    )
    input_tensors, output_tensors = export_tf_proto(args.onnx_file, args.meta_file)
    export_for_serving(args.meta_file, args.export_dir, input_tensors, output_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_path",
        help="Path to model to export.",
        type=str,
        default="models/HCapsNet.pt",
    )
    parser.add_argument(
        "--onnx-file",
        help="File where to export the ONNX file",
        type=str,
        default="tmp_onnx.onnx",
    )
    parser.add_argument(
        "--meta-file",
        help="File where to export the Tensorflow meta file",
        type=str,
        default="tf_meta.meta",
    )
    parser.add_argument(
        "--export-dir",
        help="Folder where to export proto models for TF serving",
        type=str,
        default="deployment",
    )

    args = parser.parse_args()
    main(args)
