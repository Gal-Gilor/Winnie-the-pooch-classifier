# Winnie The Pooch Classifier

## Summary

In this project, I developed a dog identification classifier. 
I used transfer learning to harness the power of the VGG16 architecture to create a dog breed classifier (133 classes multiclass classification) and a PyTorch implementation of MTCNN for face detection. The classifier achieved 74% breed classification accuracy on the test set.
The final application accepts images as input. If the image contains a dog or a face, the application tells the user which breed is the dog or which dog breed the face resembles. If the application does not detect a dog or a face, it will alert the user.
