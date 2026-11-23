## **End-to-End CNN-Based System for Human Detection in Fire Using a Flask Web Application**





This comprehensive project presents an End-to-End Convolutional Neural Network (CNN) integrated with a Flask web application for detecting humans in fire scenarios using images and video streams. The system combines a sophisticated binary fire classifier model with YOLO-based human detection to identify critical risk situations where individuals are present in fire-affected frames.



The proposed solution implements a fully automated web-based interface that enables:

* Real-time processing of images and video feeds
* Frame-by-frame analysis with advanced detection capabilities
* Live streaming of annotated video outputs with progress monitoring
* High-confidence predictions for emergency response scenarios



Our system achieves exceptional performance metrics:

* Test Accuracy: 96.26%
* Test Loss: 0.1795
* Precision on Fire Images: 96%
* Recall on Fire Images: 100%
* F1-Score: 0.95



The integrated pipeline significantly enhances early fire response capabilities, supports automated safety monitoring systems, and demonstrates remarkable accuracy in recognizing humans in fire environments. This work bridges the gap between academic research and practical deployment by providing a production-ready solution for emergency response teams and safety monitoring systems.







##### **## Installation (One time process)**

1\. Install the Anaconda Python Package and TensorFlow 2.10

 	Follow this video to understand, how to Install Anaconda and TensorFlow 2.10

 	https://youtu.be/b9e3J-NJ8TY

 

 	Important Note:

 	If TensorFlow is not running properly, downgrade the numpy version to 1.26.4 using the below command

 	>> pip install numpy==1.26.4

 

2\. Open anaconda prompt (Search for Anaconda prompt and open)



3\. Change the directory to the project folder using the below command

 	>> cd path\_of\_project\_folder

 	Example: Let's say project is present in D drive 

&nbsp;		 >>D:

 		 >>cd D:\\Detection-Of-Humans-In-Fire

 

4\. Activate the virtual environment created while installing TensorFlow using the command

 	>> conda activate tf

 

 	Note: Here tf is the virtual environment name created while installing TensorFlow

 

5\. Now install the required libraries using the below command

 	>> pip install -r requirements.txt

 





##### **## Follow the steps to train the model after installing the requirements.**



1\. Open anaconda prompt (Search for Anaconda prompt and open)



2\. Change the directory to the project folder using the below command

 	>> cd path\_of\_project\_folder

 	Example: Let's say project is present in D drive

 		 >>D:

 		 >>cd D:\\Detection-Of-Humans-In-Fire

 

3\. Activate the virtual environment created while installing TensorFlow using the command

 	>> conda activate tf



4\. Next to train the model open the Jupyter Notebook using the below command

 	>> jupyter notebook

 

5\. Open the End-to-End-CNN-for-Detection-of-Humans-in-Fire-with-Flask-Web-App.ipynb and run all cells



6\. Once the training is completed the trained model that is fire\_detection\_model.h5 will be stored in the models directory







 

##### **## Follow the steps to run the Flask Application after installing the requirements and Training the Model.**



1\. Open anaconda prompt (Search for Anaconda prompt and open)



2\. Change the directory to the project folder using the below command

 	>> cd path\_of\_project\_folder

 	Example: Let's say project is present in D drive

 		 >>D:

 		 >>cd D:\\Detection-Of-Humans-In-Fire

 

3\. Activate the virtual environment created while installing TensorFlow using the command

 	>> conda activate tf



4\. To run the Flask app and type the following command

 	>> python app.py







##### **#### Download Dataset**



Dataset Name: Fire Dataset



Dataset Link: https://www.kaggle.com/datasets/phylake1337/fire-dataset



1\. Once the dataset is downloaded, you will get an archive folder

2\. Extract the archive folder

3\. Copy and paste the fire\_dataset folder into project directory.

 



Team VTUPusle.com

