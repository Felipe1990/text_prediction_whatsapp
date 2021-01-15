import click
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from src.model import ChatBotModel

tf.get_logger().setLevel(logging.ERROR)


@click.command()
@click.option('--model_name', prompt='Which model to use?')
@click.option('--text_seed', default='te')
@click.option('--next_words', default=5)
def main(model_name: str, text_seed: list, next_words: int):
    model = ChatBotModel()

    # https://github.com/keras-team/keras/issues/10634#issuecomment-608265288
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    model.load_model_info(model_name)
    model.define(None)
    model.load_weights(model_name)
    model.predict([text_seed], next_words)

    return


if __name__ == '__main__':
    main()
