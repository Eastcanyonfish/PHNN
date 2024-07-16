# Parallel hybrid neural network (PHNN)

## Background 
The growing uncertainties of the world due to geographic tensions, weather conditions challenge the traditional method to achieve a reliable price prediction of energy future for both asset pricing and risk management. Following the initial success of deep learning models in energy price prediction, we attempt to establish a better architecture of neural networks to improve the prediction accuracy. We propose a novel parallel hybrid neural network (PHNN) model that utilizes independent sub-networks to effectively capture the distinct features of various sequences. Empirical results demonstrate that the PHNN model exhibits a significant performance enhancement of 16.68\%, 14.09\% and 2.34\% over the EMD-LSTM, the informer model and the single LSTM model, respectively. In particular, the PHNN outperforms the single LSTM, which is trained on the same inputs, by 2.34\% overall while by remarkable 4.11\% during event periods. This suggests that the PHNN derives notable benefits from its distinct architecture, particularly during the initial phase of extreme events characterized by significant price trend changes.


## File description

The folder contains five files with the following contents:

- PHNN1.py and PHNN2.py are two files dedicated to model construction, each tailored for specific operating environments.

- DataHandler.py contains code for data processing.

- evaluation.py includes code utilized for evaluating model performance.

- Demo.py showcases how to train the model and utilize the trained model for predictions.

- The data file contains a sample dataset of Natural Gas prices.

## Notes for user
The experiments in this study were conducted using the Python programming language. The computational environment was equipped with an Intel Core i5-10400F processor, 32GB of RAM, and an NVIDIA GeForce GTX 1080 graphics card. Additionally, the environment was supplemented with Python 3.8 and TensorFlow 2.10.0, a deep learning framework, to facilitate the development and testing of PHNN models.To ensure consistency and avoid conflicts with other projects, it is recommended to set up a virtual environment. This project is developed and tested using the following technologies and tools:
- Python 3.8: The primary programming language, offering robust capabilities for data processing and scientific computing.
- TensorFlow 2.10.0: A deep learning framework used for the development and testing of PHNN models. TensorFlow provides a rich set of APIs and tools that make the construction, training, and evaluation of models more efficient.
  
If you want to use the code in the old environment, you just activate the following code:

<u>from PHNN1 import PHNN_Model<u>

To facilitate the user to execute the code in upgraded environments, we also write a version for latest development environment that requests the small modifications in model configuration. we execute the code successfully both in system of Windows and MACOS with python 3.10 and tensorflow 2.17. if you want to use the code in the same environment, you just activate the following code:

<u>from PHNN2 import PHNN_Model<u>

## Citation
If you use the code from this repository please use the citation below. Additionally please cite the original authors of the models.

- @article{ZHU2024111972,
title = {Energy price prediction based on decomposed price dynamics: A parallel neural network approach},
journal = {Applied Soft Computing},
pages = {111972},
year = {2024},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2024.111972},
url = {https://www.sciencedirect.com/science/article/pii/S1568494624007464},
author = {Min Zhu and Siyue Zheng and Yu Guo and Yuping Song}
}

- M. Zhu, S. Zheng, Y. Guo et al., Energy price prediction based on decomposed price dynamics: A parallel neural network approach, Applied Soft Computing (2024),
doi: https://doi.org/10.1016/j.asoc.2024.111972.
