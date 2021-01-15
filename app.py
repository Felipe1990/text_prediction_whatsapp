import streamlit as st

from src.model import ChatBotModel

# load model
model = ChatBotModel()
results = None
st.write()

# sidebar
sidebar_header = st.sidebar.empty()
model_name = st.sidebar.selectbox('Who am I?', ['mimo', 'felipe'])

# updates based on model_name
sidebar_header.markdown(f'''
                ## Let's predict what {model_name} would say
                ''')
model.load_model_info(model_name)
model.define(None)
model.load_weights(model_name)

text_seed = st.sidebar.text_input('Starting text', value='te')
next_words = st.sidebar.number_input('How many words?', min_value=1, max_value=10, step=1, value=2, format='%.0f')

run_button = st.sidebar.button('Get prediction')

# main panel
text_description = st.empty()

if run_button:
    with st.spinner('Please wait...'):

        text_prediction = ' '.join(model.predict([text_seed], next_words))
        text_description.markdown(f'''
                ## {text_prediction} 
                ''')

else:
    text_description.markdown('''
    ## Instructions
    
    1. Select the person whose next words you'd like to predict
    2. Write an starting text to begin with
    3. Select how many words you want to predict
    4. Click on "Get prediction"
    ''')
