from logging import getLogger

import tensorflow as tf
from tensorflow.python.client import device_lib


log = getLogger(__name__)


def get_available_gpus():
    """ Get available GPU devices info. """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def test_gpu_memory_usage():
    # Detect available GPU devices info.
    log.info("On this machine, GPU devices: ", get_available_gpus())

    # Set Tensorflow GPU configuration.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    tf_config=tf.ConfigProto(
        allow_soft_placement=True,
        device_count={'GPU': len(get_available_gpus())},
        gpu_options=gpu_options,
        log_device_placement=True)
    session = tf.Session(config=tf_config)

    # Mimick training process.
    while True:
        pass
        

test_gpu_memory_usage()