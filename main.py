import streamlit as st
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.mymodels import ResNet18Based
import time

idx_to_classes_reversed = {0: '1000 back', 1: '1000 front', 2: '100 back', 3: '100 front', 4: '10 back', 5: '10 front', 6: '20 back', 7: '20 front', 8: '5000 back', 9: '5000 front', 10: '500 back', 11: '500 front', 12: '50 back', 13: '50 front'}
pth = 'checkpoints/saved_model.pth'

@st.cache
def load_model(pth: str):
    model = ResNet18Based()
    #loading pretrained results for pakistani currecy dataset
    model.load_state_dict(torch.load(pth)['state_dict'])
    model.eval()
    return model

# Function to predict the image class
def predict(img, model):
    # Define transformations
    test_transforms = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    # Apply transformations
    img_tensor = test_transforms(img)
    img_tensor.unsqueeze_(0)
    
    res = model(img_tensor)
    
    return idx_to_classes_reversed[np.argmax(res.data.cpu().numpy())]

# Load the model
model = load_model(pth)

st.title("Currency Identification App")
st.write("Upload an image and the model will identify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    with st.spinner('Identifying...'):
        labels = predict(image, model)
        time.sleep(1)
    st.success(f"Prediction: {labels}")