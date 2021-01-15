FROM tensorflow/tensorflow:2.2.2-gpu
WORKDIR /home
RUN mkdir text_prediction
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip==20.3.3 && \
    pip install -r requirements.txt