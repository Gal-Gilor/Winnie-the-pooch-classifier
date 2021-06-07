# Winnie The Pooch Classifier

## Summary

In this project, I developed a dog identification classifier.
I used transfer learning to harness the power of the [VGG16 architecture](https://neurohive.io/en/popular-networks/vgg16/) to create a dog breed classifier (133 classes multiclass classification) and a PyTorch implementation of Multi-task Cascaded Convolutional Neural Networks ([MTCNN](https://github.com/timesler/facenet-pytorch)) for face detection. The classifier achieved 74% breed classification accuracy on the test set.
The final application accepts images as input. If the image contains a dog or a face, the application tells the user which breed is the dog or which dog breed the face resembles. If the application does not detect a dog or a face, it will inform the user.

## Classification Instructions

* Clone the repository.
* Create an environment with Python 3.7.10
* Install requirements.
* [Download](https://drive.google.com/drive/folders/13n5Urxy6FbjjIV0IU0ndmS7VFVAg-guR?usp=sharing) the PyTorch model and class dictionary into the same cloned repository folder.
* Change the working directory to the newly cloned repository.
* Run the Python file with up to three image paths as arguments

## Example Usage

``` git

conda create --name dog-breeds python=3.7.10 

cd <path/to/cloned-folder>

pip install -r requirements.txt

python main.py <path/to/image>

or
 
python classify-breed.py <path/to/image> <path/to/image2> <path/to/image3>
```

## Deep Learning Models

* First, a pre-trained VGG16 detects whether there's a dog in the image. If the model prediction is between 151 and 268 (inclusive), then a dog is present. ImageNet class labels between 151 and 268 are all dog breed classes. Although VGG16 can predict dog breeds, I wanted to classify dog breed similarity on images of human faces. Since ImageNet doesn't have a specific "Human" class label, I couldn't use VGG16 out-of-the-box to meet my goal.
  
* Secondly, I use the pre-trained MTCNN to detect whether a human face is present in the picture (FaceNet). Using VGGFace2 pre-trained models, FaceNet can reach 100% accuracy on YALE, JAFFE, and AT & T datasets. FaceNet is so powerful; it also detects non-human faces with high confidence. To lower the human-face false positives rate. I decided to on a 0.97 classification confidence cut-off to reduce the false-positive rate.

* Lastly, I applied transfer learning to train my implementation of a dog breed classifier. I used the VGG16 architecture again; this time, I replaced the last 1000 neurons linear layer (classifier layer) with a 133 neurons linear layer (the number classes in my dataset). I then trained the classifier layer (I froze all the other layers' weights) for 30 epochs.

_Because the images in my train set are visually similar to the pictures on ImageNet, I decided to re-use the VGG16 architecture and the image-processing pipeline._

## Basic App Structure

* If the models detect a dog or a face in the image, I run the image through the dog breed classifier. Then, I feed the raw logits through a Softmax layer to return the probabilities.
  
* If the model is more than 65% confident about the dog breed, I classify the dog in the image as a pure breed dog. If it's less, I sort and return the two topmost probable ones.
  
* Finally, if neither is detected, the model notifies the user it cannot classify the dog breed or resemblance to one.

## Possible Improvements

* Apply transfer learning to create a human classifier model instead of the pre-trained face detector that performs too well. For example, the face detector sometimes identifies dogs' faces with high probability, similar to human faces. That's why I chose 0.975 as the cut-off point to decide whether a face is human. _Although humans are not part of ImageNet labels. Studies showed that the models detect humans as features_.
  
* Instead of returning just the original image with the model outputs for humans, I could return the original image, the resembling dog, and a mash-up between the pictures laid out side by side.

* I did not focus on interoperability in this project. However, I find it interesting to visualize the intermediate model outputs and identify what parts of the images the model uses as features to identify different breeds.
