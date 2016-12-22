# Tensor-Flow-Char-Recognition
### Background
This project aimed to teach our group about the basic working's of Google's Tensorflow and how to apply it to different datasets. For the purposes of this project, we were able to apply the code to a data similar to MNIST, called [notMNIST](http://yaroslavvb.com/upload/notMNIST/). MNIST is a dataset that is made up of numbers 0-9. notMNIST is alphabet letters A-J. We found many tutorials on the TensorFlow [website](https://www.tensorflow.org/) that detailed how to do basic machine learning with that dataset. However, much of this code we found needed to ingest the dataset in the same format that MNIST is formatted in. While notMNIST did not fit the bill perfectly, we borrowed David Flanagan's [code](https://github.com/davidflanagan/notMNIST-to-MNIST) which converts notMNIST data to the MNIST data format. From there we decided that we wanted to be able to test the code. However, the code on the TensorFlow website does not provide a method to test the model. We then found a solution on Niek Temme's [github](https://github.com/niektemme/tensorflow-mnist-predict) that allowed us to not only create the model using the basic neural network, but it also allowed us to then test the model with letters that had already been classified correctly in the notMNIST dataset, as well as our own characters that we had written in Photoshop.  
### Implementation
Setting up a development to run the code requires one in which TensorFlow is installed. Recommendations on how to do this are located [here](https://www.tensorflow.org/get_started/os_setup) on the TensorFlow website. I, however, chose to follow this [tutorial](http://www.heatonresearch.com/2016/09/10/ubuntu-tensorflow.html) which involves spinning up an Ubuntu VM and installing Anaconda for Python which allows us to create a virtual environment for running Tensorflow. Once you get to the step where you are to download TensorFlow, please change your URL to the latest version listed on the website. Afterwards you can stop the tutorial and clone our code.

### Running
Running the code is easy. First you must select which classifier you want to use (Basic Softmax, Convulutional Neural Network, Multilayer Perceptron). Navigate to that directory by going into Char-Recognition/Predict-Char-\<Classifier\>. Now you can create the model. First navigate to the create_model.py file and change the filepath to where the data resides on your computer. Run the code by typing at the command line:
```
python3 create_model.py
```

Then test the code by typing:
```
python3 predict.py <Path-To-PNG>
```
### Disclaimer
All code that was used on this repo is borrowed code from someone else. Only slight modifications have been made to existing code in terms of tweaking learning rates, optimizers, or making slight modifications to run the notMNIST data. 
