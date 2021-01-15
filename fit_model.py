import click

import tensorflow as tf

from src.model import ChatBotModel


@click.command()
@click.option('--model_name', prompt='Give the model a name')
@click.option('--participant_index', prompt='Select a participant', type=click.Choice(['0', '1']), default='0')
def main(model_name: str, participant_index):
    model = ChatBotModel()
    data = model.load_data('../data/processed/conversations.pkl')

    participant_index = int(participant_index)
    print(data.keys())
    print(list(data.keys())[participant_index])

    corpus = data[list(data.keys())[participant_index]]

    # https://github.com/keras-team/keras/issues/10634#issuecomment-608265288
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    model.define(corpus)
    model.fit()
    model.save_weights_model_info(model_name)

    return


if __name__ == '__main__':
    main()
