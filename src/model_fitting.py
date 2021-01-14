# import tensorflow
import numpy as np
import pandas as pd
import pickle


class ChatBotModel:
    model = None

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            conversations = pickle.load(f, encoding='utf-8')

        return conversations


    def define_model(self):
        return

    def fit_model(self):
        return

    def save_model(self):
        return

    def load_model(self):
        return
