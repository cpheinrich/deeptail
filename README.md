# deeptail

https://www.kaggle.com/c/whale-categorization-playground

## Getting Started
- Activate virtual environment
- pip intall -r requirements.txt
- download train.zip from https://www.kaggle.com/c/whale-categorization-playground/data (you might need to sign up for Kaggle first), unzip and rename directory to 'kaggle_train' 
- run the nameThatWhale jupyter notebook

## Image Segmentation
- An optional preprocessing step to improve classification accuracy is to first segment the whale tail from the background (typically sky and water).
- segmentation/unsupervised\ segmentation.ipynb has a pipeline for segmenting images using the open CV watershed algorithm. Unfortunately, this doesn't always deliver sastisfactory results, so we also include a few supervised segmentation approaches, and some sample training data.
- Unzip the segmentation/ directories for validation and training, and then you can run commands like in the sample_commands.txt file
- More info in segmentation/README.md


### Resources

ipynbs from Chollet's deep learning with python:
- https://github.com/fchollet/deep-learning-with-python-notebooks

Remarks: Sections 5.2 and 5.3 provide good examples of image classifiers. The main difference in the whale data is that it is not a binary classifier, and as such, we will likely want to one-hot encode the whale_ids into binary vectors. This is discussed in sections 3.6 and 6.1
