## General Idea:

1. Use [googliser](https://github.com/teracow/googliser) to download images from google, and create a dataset for training and testing.
2. Use [VGG-19 pretrained CNN](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) to extract the features of the images (here use the con5_3 layer, and the output feature shape is (512,14,14)).
3. Build the attention model and adjust the parameters.
4. Done.

## Implement

1. "Feature_Extraction" folder is used to extract features, "Preprossing" is used after the raw image is download from google, "Feature_Extra" is to extract features and encode labels, "cnn_util" contains the functions to preprocess the raw image (resize) and build CNN model with pretrained weight to extract features.

2. 
