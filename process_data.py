import src.data_processing as proc
import click


@click.command()
@click.option('--conversation_file', default='conversation')
def main(conversation_file):
    conversations = proc.read_data(f'../data/raw/{conversation_file}.txt')
    conversations_by_participants = proc.separate_conversation_by_participants(conversations)
    conversations_by_participants = proc.tidy_sentences(conversations_by_participants)

    proc.save_tidy_text(conversations_by_participants)
    return


if __name__ == '__main__':
    main()
