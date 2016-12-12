## General Idea:

1. Use [googliser](https://github.com/teracow/googliser) to download images from google, and create a dataset for training and testing.
2. Use [VGG-19 pretrained CNN](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) to extract the features of the images (here use the con5_3 layer, and the output feature shape is (512,14,14)).
3. Build the attention model and adjust the parameters.
4. Done.

## Implement

1. "Feature_Extraction" folder is used to extract features, "Preprossing" is used after the raw image is download from google, "Feature_Extra" is to extract features and encode labels, "cnn_util" contains the functions to preprocess the raw image (resize) and build CNN model with pretrained weight to extract features.

2. "Sanity_Check" folder is used to make sure the model is running good, "sanity_check.ipynb" is used to check whether the features extracted by pretrained CNN is good or not, the idea is if two location contains similar things, then they will be similar (by using cos_sim or dot_product). I also write a function to check whether the model contains enough capacity, the idea is  Allocate a fixed representation vector f_label for each of the labels. Then, for each image, instead of using as input the convolutional layer with the features a_l, set all a_l's to 0 and then randomly replace some of the a_l with the (fixed) corresponding representation of a true label of that image. The model should do very well 

3. Run "model.py" to train, and then since it also contain the testing code, so comment out train and run it again to see the result.

## Result
The result is not very good because the data I can download from google is very limited, and also the pretrained CNN is not good for some food items like rice and eggs.

![alt tag](https://github.com/1230pitchanqw/Food_image_recognition/blob/master/result/11.png)\\
![alt tag](https://github.com/1230pitchanqw/Food_image_recognition/blob/master/result/12.png)



Note that I aslo ran a similar model and have a better result. This work is done using a published dataset-- Food101, with 101 categories and 101,000 food images.\\

![alt tag](https://github.com/1230pitchanqw/Food_image_recognition/blob/master/result/21.png)\\
![alt tag](https://github.com/1230pitchanqw/Food_image_recognition/blob/master/result/22.png)
