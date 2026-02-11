&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/OverviewFigureCrop.png" width = "500" />
# Sensor Response Time Reduction

***
For full details, see the following publications:

Ward, S. J., Baljevic, M., & Weiss, S. M. (2024). Sensor Response-Time Reduction using Long-Short Term Memory Network Forecasting. _arXiv_ 2404.17144. doi: [10.48550/arXiv.2404.17144](https://doi.org/10.48550/arXiv.2404.17144)

Ward, S. J., & Weiss, S. M. (2023). Reduction in sensor response time using long short-term memory network forecasting. _Proc. SPIE_, __12675__(126750E). doi: [10.1117/12.2676836](https://doi.org/10.1117/12.2676836)

***
## Table of Contents
### 1. Motivation
### 2. Experimental Data
#### 2.1 Porous Silicon
#### 2.2 Data Collection
#### 2.3 Data Visualization
&emsp;&emsp; 2.3.1 Full Dataset

&emsp;&emsp; 2.3.2 Equilibrium Sensor Response vs Protein Solution Concentration
#### 2.4 Model Hyperparameter Tuning
#### 2.5 Model Training
#### 2.6 Model Evaluation
&emsp;&emsp; 2.6.1 Model Architecture

&emsp;&emsp; 2.6.2 Representative Examples of Model Predictions

&emsp;&emsp; 2.6.3 Histogram Showing Prediction Response Time Improvement

&emsp;&emsp; 2.6.4 Box and Whisker Plot Showing Prediction Response Time Improvement

&emsp;&emsp; 2.6.5 Comparison of Ensemble Sizes

&emsp;&emsp; 2.5.6 All Model Predictions
### 3. Simulated Data
### 4. Alternative Models
### 5. Troubleshooting
### 6. FAQ
***
## 1. Motivation

The response time of a biosensor is a crucial metric in safety-critical applications such as medical diagnostics where an earlier diagnosis can markedly improve patient outcomes. However, the speed at which a biosensor reaches a final equilibrium state can be limited by poor mass transport and long molecular diffusion times that increase the time it takes target molecules to reach the active sensing region of a biosensor.

While optimization of system and sensor design can promote molecules reaching the sensing element faster, a simpler and complementary approach for response time reduction that is widely applicable across all sensor platforms is to use time-series forecasting to predict the ultimate steady-state sensor response.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "/Figures/StrategyAnimation.gif" width = "600" />

In this work, we show that ensembles of long short-term memory (LSTM) networks can accurately predict equilibrium biosensor response from a small quantity of initial time-dependent biosensor measurements, allowing for __significant reduction in response time by a mean and median factor of improvement of 18.6 and 5.1, respectively__. The ensemble of models also provides simultaneous estimation of uncertainty, which is vital to provide confidence in the predictions and subsequent safety-related decisions that are made.

This approach is demonstrated on real-time experimental data collected by exposing porous silicon biosensors to buffered protein solutions using a multi-channel fluidic cell that enables the automated measurement of 100 porous silicon biosensors in parallel. The dramatic improvement in sensor response time achieved using LSTM network ensembles and associated uncertainty quantification opens the door to trustworthy and faster responding biosensors, enabling more rapid medical diagnostics for improved patient outcomes and healthcare access, as well as quicker identification of toxins in food and the environment.
***
## 2. Experimental Data
### 2.1 Porous Silicon
Porous Silicon (PSi) is silicon with nanostructured pores, which have been electrochemically etched using hydrofluoric acid. Below are shown some images on the scale of a few nanometres taken using an electron microscope, and and illustration of how molecules are captured and detected optically in the pores.

![](https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/PorousSilicon.gif)

### 2.2 Data Collection
Porous silicon sensors were fabricated, secured in a multi-channel fluidic cell, and real time optical reflectance measurements were carried out for each sensor in turn as the protein bovine serum albumin (BSA) in buffer solutions (HEPES), at concentrations of 40, 20, 10, 4, 2, 1, 0.4, 0.2, 0.1, 0.04, 0.02, 0.002 mg/ml, and a control solution consisting of 100% buffer, were dropped onto the sensors.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "/Figures/BSA.png" width = "400" /> 

Collection of a sufficiently large dataset was enabled by using an automated real-time measurement setup in which the multi-channel fluidic cell was affixed to a stepper motor, which was controlled using python to cycle through the reflectance measurement of many sensors in sequence.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "/Figures/HighThroughputMeasurementSetup.png" width = "700" />
***
### 2.3 Data Visualization
#### 2.3.1 Full Dataset
The full experimental dataset, consisting of 387 examples of time series sensor response data:

<img src = "/Figures/ExperimentalTrainingDataset.png" width = "300" />

The same plot but only showing a representative sample of protein concentrations:

<img src = "https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/RepresentativeExperimentalTrainingDataset.png" width = "300" />

#### 2.3.2 Equilibrium Sensor Response vs Protein Solution Concentration
The adsorption isotherm, which refers to the variation of equilibrium sensor response with protein concentration:

<img src = "https://github.com/SimonJWard/Response-Time-Reduction/blob/main/Figures/SemilogAverageEquilibriumResponse.png" width = "300" />

***
## 2.4 Model Hyperparameter Tuning

The architecture resulting in minimum loss on the validation dataset was explored through hyperparameter tuning with Keras tuner.

## 2.5 Model Training

An ensemble of 15 base learners were trained using the optimal hyperparameters.

## 2.6 Model Evaluation
### 2.6.1 Model Architecture
LSTM networks are well suited for the rapid prediction of equilibrium sensor response due to their ability to learn features without requiring manual feature selection, to learn to distinguish signal from noise, and to learn long and short term dependencies in sequential data, all of which promote generalizability. 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src = "/Figures/LSTM.png" width = "700" />

Models were implemented in tensorflow using the keras API and built-in LSTM layers. Each LSTM layer was configured to return a sequence of 250 outputs, one for each input time step. The target output, used to compute the loss, is the final element of each input sequence, repeated in a vector with 250 elements. Each output prediction in the sequence is made having only seen data from the current and past timesteps, so will typically become increasingly more accurate as the sequence goes on and more data from the input sequence is seen by the model.

The data was first randomly shuffled and split into train, validation and test sets at a ratio of 3:1:1, stratified by BSA concentration. Ensembles of 15 base learners were trained in turn, by minimizing the negative log likelihood (
−log p(y|x) ), using softplus activation at the output layer to ensure predictions are positive, and adam optimization. Ensembles were used to increase accuracy and prediction stability, and for better calibrated uncertainty quantification.

The base learner architecture, informed by limited hyperparameter tuning using the validation set, was the following: 50 input neurons, 1 hidden layer with 500 neurons, and 2 output neurons. The maximum and minimum sensor response values across all time steps and all examples in the training set were used to normalize the train, validation and test sets, to avoid data leakage.
***
### 2.6.2 Representative Examples of Model Predictions

Six examples of predictions made on experimental sensor response time series, individually and in a panel:

<img src = "/Figures/PanelExperimentalEnsemblePredictionIdealResponses.png" width = "400" />

***
### 2.6.3 Histogram Showing Prediction Response Time Improvement 

A histogram comparing the distributions of response times for the unprocessed experimental data and model predictions, including normalized variance which is indicative of signal to noise ratio (S/N):

<img src = "/Figures/PredictedExperimentalt90HistogramNormVariance.png" width = "300" />

***
### 2.6.4 Box and Whisker Plot Showing Prediction Response Time Improvement 

A box and whisker plot comparing the distributions of response times for the unprocessed experimental data and model predictions:

<img src = "/Figures/BoxandWhiskerPlotExperimentalPredictedt90Ratio.png" width = "300" />

***
### 2.6.5 Comparison of Ensemble Sizes 

The mean and median improvement in response time for different numbers of base learners (1-30):

<img src = "/Figures/EnsembleSizeResponseTimeImprovement.png" width = "300" />

### 2.5.6 All Model Predictions 

Predictions made on all experimental sensor response time series in the test set.

## 6. FAQs

***
## 7. Acknowledgements

***
