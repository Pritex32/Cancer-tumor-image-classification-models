import streamlit as st
from streamlit_option_menu import option_menu
from keras.models import load_model
import tensorflow
import numpy as np
from PIL import Image

brain=load_model('brain_tumor_image_model3.h5')
kidney=load_model('kideney_stone_cnn_model3.h5')
lungs=load_model('lungs cancer cnn model3.h5')

with st.sidebar:
    selected=option_menu(menu_title='medical scan diagnosis'.upper(),
                         options=['brain','kidney','lungs'],
                         icons=['Arrow up-right','Arrow up-right','Arrow up-right'],
                         default_index=0)
    

if (selected=='brain'):
    st.title('brain tumor scan diagnosis'.upper())
    name=st.text_input('please input your name:')
    if name:
        st.write(f"**Welcome {name}!** Please upload your file.")

    uploaded=st.file_uploader('upload your scan',type=['jpg','jpeg','png'])
    if uploaded is not None:
        img=Image.open(uploaded)
        img_size=img.resize((100,100))
        img_array=np.array(img_size)
        img_dm=np.expand_dims(img_array,axis=0)
        img_norm=img_dm/255.0

        st.image(uploaded ,caption='uploaded image')
        info =''

        if st.button('classify'):
            prediction=brain.predict(img_norm)
            pred_arg=np.argmax(prediction,axis=1)[0]
            class_names=['Healthy', 'Tumor']
            pred=class_names[pred_arg]
            st.success(pred)

if (selected=='kidney'):
    st.title('kidney stone scan diagnosis'.upper())
    st.write('welcome')

    uploaded=st.file_uploader('upload your scan',type=['jpg','jpeg','png'])
    if uploaded is not None:
        img=Image.open(uploaded)
        img_size=img.resize((100,100))
        img_array=np.array(img_size)
        img_dm=np.expand_dims(img_array,axis=0)
        img_norm=img_dm/255.0

        st.image(uploaded ,caption='uploaded image')

        if st.button('classify'):
            prediction1=kidney.predict(img_norm)
            pred_arg=np.argmax(prediction1,axis=1)[0]
            class_names={0:'Normal' ,1:'stone'}
            pred=class_names[pred_arg]
            st.success(pred)



if (selected=='lungs'):
    st.title('lungs cancer scan diagnosis'.upper())
    st.write('welcome')

    uploaded=st.file_uploader('upload your scan',type=['jpg','jpeg','png'])
    if uploaded is not None:
        img=Image.open(uploaded)
        img_size=img.resize((100,100))
        img_array=np.array(img_size)
        img_dm=np.expand_dims(img_array,axis=0)
        img_norm=img_dm/255.0

        st.image(uploaded ,caption='uploaded image')

        if st.button('classify'):
            prediction=lungs.predict(img_norm)
            pred_arg=np.argmax(prediction,axis=1)[0]
            class_names=['adenocarcinoma','benign', 'squamous_cell_carcinoma']
            pred_class=class_names[pred_arg]
            st.success(pred_class)
