# Predicting the Probability of a Heart Attack

## Introduction
The main objective of this project is to develop and deploy a predictive model that is capable of estimating the likelihood of a patient experiencing a heart attack. Eventually, the model's prediction will be compared to actual results to assess its accuracy and validity.

Alongside the validation process, the overarching goal is to develop expertise in model training, allowing for further creation of more robust and effective predictive models in the future


## Selection of Data
The model processing and training are conducted using a Colab Notebook and is available [here](codes/Predictive_Model.ipynb)

The data contains more that 300 samples of patient data, including these features:
- age: Age of patient
- sex: Sex of patent
- cp: Chest Pain type
	- 1: typical angina
	- 2: atypical angina
	- 3: non-anginal pain
	- 4: symptomatic
- trtbps: Resting blood pressure (in mm Hg)
- chol: Cholesterol in mg/dl fetched via BMI sensor
- fbs: fasting blood sugar > 120 mg/dl
	- 1: True
	- 0: False
- restecg: Resting electrocardiographic results
	- 0: normal
	- 1: having ST-T wave abnormality (T wave inversion and/or ST elevation or depression of > 0.05 mV
	- 2: showing probable or definite left ventricular hypertrophy by Estes’ criteria
- thalachh: maximum heart rate achieved
- exang: Exercise induced angina
	- 1: yes
	- 0: no
- oldpeak: previous peak
- slp: slope
- caa: Number of major vessels (0-3)
- thall: thal rate
- output
	- 0: less chance of heart attack
	- 1: more chance of heart attack

Data preview:
[images/data_preview.png]

Normally for data there are both categorical and numerical features, therefore both types have to be handled in different ways, but since the data I use only contains numerical features, I only had to use a normalization layer.
[graph/normalization_layer.png]

## Methods
Tools:
- NumPy, Matplotlib, Seaborn, Pandas, TensorFlow
- GitHub for hosting

Resources:
- TensorFlow Tutorials
- MIT Introduction to Deep Learning 6.S191

Functions:
 - Normalization Layer
 - Dataframe to Dataset

## Results
Prediction model can be found [here](data/heartattackpred.keras)
Which can be loaded with the following:
```
import tensorflow as tf

load_model = tf.keras.models.load_model(‘heartattackpred.keras’)
```

## Discussion and Summary
When initially examining the pair plot of the features for each patient, it’s hard to discern any clear patterns to determine whether a patient has a higher risk of experiencing a heart attack. However, in the world of machine learning models, they are able to identify things we can’t. Models possess an incredible capacity to go through complex datasets and uncover any details and patterns that aren’t clear to us. Learning to construct such a model has helped me understand how helpful machine learning can be. It’s not just about choosing a specific algorithm for the model, but how to select relevant features, fit the model to said features, and interpret the model’s predictions. This mindset has allowed me to understand the intricacies of making machine learning models and the field of predictive modeling.

References

[1] [Kaggle dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

[2] [TensorFlow](https://www.tensorflow.org/)

