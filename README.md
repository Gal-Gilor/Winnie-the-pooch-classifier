# Winnie The Pooch Classifier

## Summary

In this project, I developed a dog identification classifier.
I used transfer learning to harness the power of the [VGG16 architecture](https://neurohive.io/en/popular-networks/vgg16/) to create a dog breed classifier (133 classes multiclass classification) and a [PyTorch implementation of MTCNN](https://github.com/timesler/facenet-pytorch) for face detection. The classifier achieved 74% breed classification accuracy on the test set.
The final application accepts images as input. If the image contains a dog or a face, the application tells the user which breed is the dog or which dog breed the face resembles. If the application does not detect a dog or a face, it will inform the user.

## Classification Instructions:

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
 
python main.py <path/to/image> python main.py <path/to/image2> python main.py <path/to/image3>
```
