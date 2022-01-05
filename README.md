# Regression_CapsNet
Implementation of a regression capsule network for use in li-ion battery RUL prediction

This repository was made for the paper: 'Novel Image-Based Rapid RUL Prediction for Li-Ion Battery using Capsule Network' 
Author: Jonathan Couture, OntarioTech University
E-mail: jonathan.couture@ontariotechu.net
Date: August 29th 2021

In this, the Toyota/MIT dataset was used to create various datasets representing consecutive data cycles shown to the network to then train the network to predict
the remaining useful life of the battery that it sees. The images were preprocessed and created using Matlab.
5 inputs are used on the image, the current curves for a charging and discharging cycle, the charging capacity over time, the discharging capacity over time,
and then the final discharging capacity at the end of the nth cycle along with the internal resistance (both normalized) are added as numerical data to the image.

The curve in red represents the first cycle, and the curve in blue represents the final cycle, n, (3, 5, or 10). The accuracy of the model is then evaluated for the different
scenarios and also compared to a single cycle (only one curve shown). 

If you'd like to try out the network with the data, you'd need to download the Toyota/MIT battery dataset from the source at:
https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
and run the Matlab scripts
