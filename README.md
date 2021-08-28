# Reg_CapsNet
Implementation of a regression capsule network for use in li-ion battery RUL

This repository was made for the paper: 'Novel Image-Based Rapid RUL Prediction for Li-Ion Battery using Capsule Network'
Author: Jonathan Couture, OntarioTech University
email: jonathan.couture@ontariotechu.net

In this I use the Toyota/MIT dataset to create various datasets representing consecutive data cycles shown to the network, and then train the network to predict
the remaining useful life of the battery that it sees. The images were preprocessed and created using Matlab.
5 inputs are used on the image, we see the current curves for a charging and discharging cycle, the charging capacity over time, the discharging capacity over time,
and then the final discharging capacity at the end of the nth cycle along with the internal resistance (both normalized) are added as numerical data to the image.

The curve in red represents the first cycle, and the curve in blue represents the final cycle, n, (3, 5, or 10). We evaluate the accuracy of the model for the different
scenarios and also compare it to a single cycle (only one curve shown). 
