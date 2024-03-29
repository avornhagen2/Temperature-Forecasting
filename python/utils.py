import tensorflow as tf
import numpy as np


# convert an array of values into a dataset matrix


def create_datasetLSTM(dataset, look_back=10, yearsForTest=1):
    trainX, trainY = [], []
    testX, testY = [], []
    for i in range(len(dataset) - look_back - 1 - 365*yearsForTest):

        a = dataset[i:(i + look_back)]
        trainX.append(a)
        trainY.append(dataset[i + look_back])
    print("Start of test: %d" % (len(trainX)))
    for i in range(len(trainX), len(dataset) - look_back - 1):

        a = dataset[i:(i + look_back)]
        testX.append(a)
        testY.append(dataset[i + look_back])
    return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory
