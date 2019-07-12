

Use python-3 and pytorch library. 
Anaconda 3.7 Cuda 10

To use colab add the dataset file to the drive and write :

from google.colab import drive
drive.mount('/content/drive')

and give the path of the dataset something like "drive/My Drive/dataset"

We have 263 training, 25 validation and 25 test videos and related annotation files.


In this task, we made a single object tracking with regression networks. We take the images as pairs and we crop them according to bounding boxes and find the relative positions of the bounding boxes.
Then we extract features using the VGG-16 model from both frames in each pairs and then concatanete them. After we train a basic fully connected neural network, we test the network and create 3 different 
videos from 3 different classes to see the results.

The classes and functon as below:

class Network(nn.Module) => # This is a class structure which construct the structure of the basic fully connected neural network.

class GetFeatures(Dataset) => # This is a class structure which is the custom loader to take the features and the boxes as batches. Takes the array of features and the array of boxes as parameter.
# Returns the features and relative boxes.

class GetImages(Dataset) => # This is a class structure which is the custom loader to take the images as pairs. 
# To call custom loader that takes images as pairs give the csv files as dataframe, the path of the folder that has the images, transform value and a boolean value if this is runned on Google Colab or not.
# Returns the image pairs, name of the video and the relative bounding boxes respectively.

def ann_file(path, isColab = True) => # Returns the coordinates of the frames as dictionary.
# Takes 2 parameters which are path of the folder and a boolean value if the code is running on Google Colab or not

def csv_files(path, ann, isTrain=True, isTest=False, isColab = True) => # creates csv extension files which contains the frame1_path, frame2_path, bounding_box_of_frame1 and bounding_box_of_frame2 respectively in each line
# Takes path of the videos folder, dictionary of the annotations that is returned from ann_file function, and 3 boolean values respectively
# make isTrain true if want to take train images
# make isTest true and isTrain false if want to take test images
# make isTest false and isTrain false if want to take validation images
# make isColab true if the code running on Google Colab otherwise false

def enlarged(im, box) => # makes the bounding box 2 times large and return the coordinates of the enlarged box as an array
# takes the image and the ground truth bounding box as parameter

def relative(enlarge, box) => # finds the relative position of the bounding box and returns its coordinates as an array
# takes the coordinates of the enlarged box and the ground truth bounding box of the second frame

def scaled(enl, box) => # reorganize the coordinates of the bounding box according to the scaling and return  its coordinates as an array
# takes coordinates of the enlarge box and the coordinates of the relative box

def crop_im(im, box1, box2) => # returns the coordnates of the enlarged and relative boxes. 
# takes image, ground truth bounding box of the first frame and ground truth bounding box of the second frame as parameter

def get_class(img_path) => # takes the name of the video from the given path and return it
# takes the path of the image as parameter

def feature_extract(model, dataloader, isTrain) => # extracts the features from the model and return the combination of the feaatures and the boxes
# takes model, dataloader and a boolean value if it is a train dataset or not

def train(net, device, tr_dataloader, val_dataloader, optimizer, criterion, epoch_num) => # Trains the fully connected neural network and returns the network, loss of the validation and loss of the train
# Takes the model, the device(gpu or not), train dataloader, validation dataloader, optimizer, criterion and the number of epochs as parameter

def plot(val_loss, tr_loss, num_epochs) => # draw the plot of losses annd save them
# takes array of validation losses, array of train losses and number of epochs as parameter

def colab_path(path, isColab = True) => # prepare the given path according to running environment
# if it is run in colab, make the isColab = True (default = True)



def test(csv_test, net) => # take test dataframe and the network 
# return the average loss
# test the network






































	
	
	