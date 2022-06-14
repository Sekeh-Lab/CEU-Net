# CEU-Net for Hyperspectral Image Semantic Segmentation

By: Nicholas Soucy and Dr. Salimeh Yasaei Sekeh

Offical repository for the CEU-Net code from the paper titled: "CEU-Net: Ensemble Semantic Segmentation of Hyperspectral Images Using Clustering" by Nicholas Soucy and Salimeh Yasaei Sekeh.

## Description

Most semantic segmentation approaches of  hyperspectral images use and require preprocessing steps in the form of patching to accurately classify diversified land cover in remotely sensed images. These approaches use patching to incorporate the rich spacial neighborhood information in images and exploit the simplicity and segmentability of the most common datasets. In contrast, most landmasses in the world consist of overlapping and diffused classes, making neighborhood information weaker than what is seen in common datasets. To combat this common issue and generalize the segmentation models to more complex and diverse hyperspectral datasets, in this work, we propose a novel flagship model: Clustering Ensemble U-Net. Our model uses the ensemble method to combine spectral information extracted from convolutional neural network training on a cluster of landscape pixels.

Clustering Ensemble U-Net (CEU-Net) works by leveraging clustering and ensemble methods to create subsets of the data so each sub-U-Net within the ensemble model becomes an expert at a clustered subset of the data.


## Prerequisites

* [Scikit-Learn](https://scikit-learn.org/stable/install.html): 1.0.2
* [Numpy](https://numpy.org/install/): 1.20.3
* [Spectral](https://www.spectralpython.net/installation.html): 0.22.4
* [Tensorflow](https://www.tensorflow.org/install): 2.3.0
* [Keras](https://keras.io/getting_started/): 2.4.3

## Model

In this paper we develop two deep learning semantic segmentation methods: Single U-Net and CEU-Net. Both are found in the "networks.py" script.

<img src="Figure Examples/CE U-Net.png"/>

## Directions

The most direct visualization of the code is provided in "CEU-Net-Example.ipynb." For running the code more efficiently, I recommend running via "run.py."

## Results

Results can be found in the paper. In addition, some of the visual results are provided in the folder "Figure Examples."

## Acknowledgements

Original Single U-Net code was based of the github by [thatbrguy](https://github.com/thatbrguy/Hyperspectral-Image-Segmentation)

Changes to the code include different preprocessing and layer parameters.
