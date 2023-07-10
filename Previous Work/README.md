# Training-Private-Deep-Learning-Models-for-Medical-Image-Analysis

In this project, we developed a state-of-the-art differentially private neural network for binary classification of chest radiography images to detect pneumonia. The dataset used in this project can be downloaded from the provided link.

https://drive.google.com/drive/folders/1yIP88m1oUjeTNae7o4mCfj86ZHcA02mP

To reproduce our results, please follow the steps below:

$\bullet$  Run the "Data Preprocessing.ipynb" script. Make sure to have the "archive.zip" file saved in the working directory before running this script. This script will preprocess the dataset and prepare it for training.

$\bullet$   Next, run the "Final - Optimized Script for MedMnist - Binary Classification - Membership Attack (TF 2.5).ipynb" script. This script contains the optimized implementation of the deep learning model using TensorFlow 2.5. It will train the model using the preprocessed data and perform binary classification on the chest radiography images.

$\bullet$  After running the model, you can visualize the results by running the "Visualizations.ipynb" script. This script reads the results stored in the "Data.csv" file, which is available for download from the provided link. It provides visualizations and insights based on the model's performance.

By following these steps, you will be able to reproduce our work and gain a better understanding of our approach in training private deep learning models for medical image analysis.
