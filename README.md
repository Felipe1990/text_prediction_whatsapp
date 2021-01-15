# Text completion using whatsapp conversations

## How to?

### 1. Prepare the data

Run `process_data` passing the name of the conversation file, that should be store in the folder `data/raw`

### 2. Fit the model

#### 2.1 CPU

1. Create the conda environment using the file `env_whatsapp`
2. Comment the lines 22 - 26  in the script `fit_model`
3. In the environment run `fit_model` passing as argument the name of the model and the index of the participant which sentences will be used
```
python -m fit_model
```

#### 2.2 GPU

1. Create an image container using the instructions from `Dockerfile`
2. Spin a container using the created image mounting the volumes where the data and code are:
```
sudo docker run --gpus all -v /home/felipe/Documents/side_projects/whatsapp_chatbot/data:/home/text_prediction/data \
                           -v /home/felipe/Documents/side_projects/whatsapp_chatbot/code:/home/text_prediction/code -it code_tf bash
```
3. Go to the folder `/home/text_prediction/code` and run `fit_model` passing as argument the name of the model and the index of the participant which sentences will be used

### 3. Use the model

#### 3.1 Command line

From the command line run `predict` passing as argument the model name (previously fitted), the text seed and the number of words that you want to predict.

#### 3.2 streaml

1. From the folder code run`streamlit run app.py`
2. In your browser go to: [http://localhost:8501/]()