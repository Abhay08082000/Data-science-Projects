import streamlit as st
from PIL import Image
import pickle
from img2vec_pytorch import Img2Vec
import torchvision.transforms as transforms
img2vec = Img2Vec()



def load_model(path):
    model=pickle.load(open(path,"rb"))
    return model
SVC=load_model(r"C:\Users\bisht\OneDrive\Desktop\Family guy\Family Guy Image Classifier\sv.pkl")
st.write("""
         # FamilyGuy Image Classification
         """
         )
file=st.file_uploader('Please upload an character image',type=['jpg','png'])
import cv2 as cv
from PIL import Image,ImageOps
transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to a specific size
        transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
    ])

def image_and_predict(image_data,model):

    img = transform(image_data)
    img_pil = transforms.functional.to_pil_image(img)
    img_features = img2vec.get_vec(img_pil)
    prediction=model.predict([img_features])
    return prediction
if file is None:
    st.text("Please upload the image file")
else:
    image=Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)
    prediction=image_and_predict(image,SVC)
    pred=prediction[0]
    st.success(f'the family guy charecter name is {pred}')





