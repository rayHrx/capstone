import tensorflow as tf
import numpy as np

from capstone import model_function
from capstone import inputs

def main():
    mnist = tf.contrib.learn.datasets.load_dataset()
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir="/tmp/convnet_model",
                                        params={
                                            'learning_rate': 0.001
                                            'n_classes': 10
                                        })

    rensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensor=tensor_to_log, every_n_iter)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[])
