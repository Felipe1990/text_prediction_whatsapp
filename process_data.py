import src.data_processing as proc
import click


@click.command()
def main():
    conversations = proc.read_data('../data/raw/conversation.txt')
    conversations_by_participants = proc.separate_conversation_by_participants(conversations)
    conversations_by_participants = proc.tidy_sentences(conversations_by_participants)

    proc.save_tidy_text(conversations_by_participants)
    return


if __name__ == '__main__':
    main()
