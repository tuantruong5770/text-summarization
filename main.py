import tensorflow as tf


def test_built_with_gpu():
    print("Tensorflow built with CUDA:", tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    pass

