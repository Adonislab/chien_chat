import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image



@st.cache(allow_output_mutation=True)
def load():
    model_path = "cat_dog.h5"
    model = load_model(model_path, compile=False)
    return model

# Chargement du model
model = load()


# Ajout des informations sur le nom, le prénom, le lien et l'hommage pour le professeur
st.sidebar.title("Informations")
nom = "NOBIME"
prenom = "Tanguy Adonis"
whatssap = "+22951518759"
lien = "www.linkedin.com/in/tanguy-adonis-nobime-078166200"

def predict(image):
    img = Image.open(image)
    img = np.asarray(img)
    print(img.shape)
    img_resize = np.array(Image.fromarray(img).resize((150, 150)))
    img_resize = np.expand_dims(img_resize, axis=0)
    print(model)
    pred = model.predict(img_resize)
    rec = pred[0][0]
    print(rec)
    return rec

st.title("Fameux : Reconnaître un chat et un chien")

upload = st.file_uploader("Chargez l'image de votre photo", type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
    rec = predict(upload)
  

    c1.image(Image.open(upload))
    if rec ==1:
        c2.write(f"Il semble que c'est un chien")
    else:
        c2.write(f"Il semble que c'est un chat")

# Affichage des informations dans la barre latérale
st.sidebar.write(f"Nom: {nom}")
st.sidebar.write(f"Prénom: {prenom}")
st.sidebar.write(f"Linked: {lien}")
