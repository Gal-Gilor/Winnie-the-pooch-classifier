import os
from glob import glob
from PIL import ImageFile, Image
import sys
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from facenet_pytorch import MTCNN

import streamlit as st


def face_detector(image_path: str) -> float:
    '''
    using pretrained VGG_FACE2 model to detect faces
    '''
    # check if CUDA is available
    cuda = torch.cuda.is_available()

    # load image
    image = Image.open(image_path).convert('RGB')  # ensure color image

    # create a face detection pipeline using MTCNN (suggested parameters):
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True
    )  # model trained on image 160 pixel size images

    _, proba = mtcnn(image, return_prob=True)
    try:
        return round(proba, 6)
    except:
        return 0


def dog_detector(image_path: str) -> bool:
    '''
    using pretrained VGG model to identify images containing dogs
    '''

    # load image
    image = prepare_image(image_path)

    # load pretrained ImageNet model
    VGG16 = models.vgg16(pretrained=True)

    # check if CUDA is available
    cuda = torch.cuda.is_available()

    # move model to GPU if CUDA is available
    VGG16.eval()
    with torch.no_grad():
        if cuda:
            VGG16 = VGG16.cuda()
            image = image.cuda()

        outputs = VGG16(image)

    # return the index for largest value
    pred = outputs.data.to('cpu').numpy().argmax()
    # check whether the image contains a dog
    is_dog = True if pred >= 151 and pred <= 268 else False

    return is_dog


def prepare_image(image_path: str) -> torch.Tensor:
    '''
    using pytorch transforms to load and procces image for dog breed classification
    '''

    # load image
    image = Image.open(image_path)  # .convert('RGB')  # ensure color image

    # process image for vgg 16
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    img = transform(image)  # apply transforms
    img = img[None, ...].float()  # account for batch size

    # ensure prepare_image returns the correct data type and shape
    assert type(img) == torch.Tensor, 'The function prepare_image() does not return\
    the expected data type. It is suppose to convert a PIL image to Tensor'

    assert img.shape == torch.Size([1, 3, 224, 224]), 'The function prepare_image() does not return\
    the expected data type. Expected shape [1, 3, 224, 224] - [batch, channels, height, width]'

    return img


def run_app(img_path: str):

    # assert type(img_path) == str, 'Function requires a image path as a string'

    # check whether the image contains an image of a dog
    is_dog = dog_detector(img_path)

    # check whether the image contains a face
    face = True if face_detector(img_path) > 0.975 else False

    # check if CUDA is available
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        # load dog breed classifier
        dog_breed_labels = jb.load('class_names.pkl')
        dog_breed_model = torch.load('model_transfer.pt')

        # prepare image for breed classification
        image_tensor = prepare_image(img_path)

        # classify the breed
        if is_dog or face:
            dog_breed_model.eval()  # ensure evaluation mode
            with torch.no_grad():
                dog_breed_model = dog_breed_model.to(device)
                image_tensor = image_tensor.to(device)
                output = dog_breed_model(image_tensor)

            # return probabilities
            softmax = nn.Softmax(dim=1)
            probailities = softmax(output)

            # extract top breeds
            n_breeds = 2
            proba, ind = torch.topk(probailities, n_breeds)
            proba, ind = proba.squeeze(), ind.squeeze()

            top_dogs = {dog_breed_labels[ind[i].item()]: round(
                proba[i].item()*100, 2) for i in range(n_breeds)}

            # compose user-message
            breeds = list(top_dogs.keys())
            probas = list(top_dogs.values())

        # find if there's a dog in the image
        if is_dog:

            # if the classifier is more than 65% sure about a single breed, call it pure bread
            if probas[0] >= 65:  # random cut-off
                message = f'The dog in the image looks like a pure-bread {breeds[0]}'

            else:
                message = f'''The dog in the image seems to be at least {probas[0]}% {breeds[0]}\
                \nand {probas[1]}% {breeds[1]}'''

        # check if there's dog but there is a face
        elif face:
            message = f'''The person in the image resembles {probas[0]}% {breeds[0]}\
            \nand {probas[1]}% {breeds[1]}'''

        else:
            message = 'Unable to classify the dog breed. Please try a different image'

    except Exception as e:
        print(str(e))
        message = 'Unable to classify the dog breed. Please try a different image'

    return message


##############################
########## Frontend ##########
##############################


page = st.selectbox("Winnie the Pooch - Page Selection", [
                    "Welcome Page", "Inference Page"])

if page == 'Welcome Page':
    st.markdown("<h1 style='text-align: center;'>Welcome!</h1>",
                unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center;'>Are you ready to find out what dog breed you resemble?!</h4>",
                unsafe_allow_html=True)

    st.markdown('''
        In this project, I developed a dog breed classifier.
        I used transfer learning to harness the power of the [VGG16 architecture](https://neurohive.io/en/popular-networks/vgg16/) to create a dog breed classifier (133 classes multiclass classification)
        and a PyTorch implementation of Multi-task Cascaded Convolutional Neural Networks ([MTCNN](https://github.com/timesler/facenet-pytorch)) for face detection.
        The classifier achieved 74% accuracy on the test set and
        is ready to tell you which dog breed you resemble!
    ''')

    st.markdown('''
        
    ''')

    st.markdown("<h2 style='text-align: center;'>Instructions</h1>",
                unsafe_allow_html=True)
    st.markdown('''
        1. Click the selection box above, and press the "Inference Page" option.
        2. Click the "Browse files" button to choose an image located on your local device.
        3. Sit back, relax, and let the model do the heavy lifting!
    ''')

else:
    uploaded_image = st.file_uploader(
        "Upload Image", type=['png', 'jpeg', 'jpg'])

    if uploaded_image is not None:
        st.image(uploaded_image, use_column_width=True)
        predict = run_app(uploaded_image)
        st.write(predict)

    else:
        st.markdown("<h2 style='text-align: center;'>Upload a picture of yourself to see which breed you resemble!</h2>",
                    unsafe_allow_html=True)
        st.write("* _Supported Formats: PNG, JPG, JPEG_")
