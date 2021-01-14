# from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


def read_data(file_path: str):
    conversations = []
    with open(file_path, 'r') as f:
        for line_i in f:
            conversations.append(line_i)
    return conversations


def separate_conversation_by_participants(conversations: list):
    separated_conversation = {}
    for entry_i in conversations:
        try:
            participant = entry_i.split(':')[1].split('-')[1].strip()
            message = entry_i.split(':')[2].strip().lower()
            # print(f'{participant}: {message}')
            if participant not in separated_conversation.keys():
                separated_conversation[participant] = [message]
            else:
                separated_conversation[participant].append(message)
        except IndexError:
            pass

    # print(separated_conversation)
    return separated_conversation


def tidy_sentences(conversations: dict):
    adj_conversations = conversations
    for participant_i in conversations.keys():
        adj_conversations[participant_i] = [sentence for sentence in
                                            adj_conversations[participant_i] if
                                            sentence != '<multimedia omitido>']
    return adj_conversations


def save_tidy_text(conversations: dict,
                   file_name: str = 'conversations'):
    with open(f'../data/processed/{file_name}.pkl', 'wb') as f:
        pickle.dump(conversations, f)
