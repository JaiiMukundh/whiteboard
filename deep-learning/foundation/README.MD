# Deep Learning Foundations: Concept Overview

This repository contains presentations and conceptual overviews covering the evolution of deep learning, from basic artificial neurons to advanced sequence models.

## 1. The Building Blocks
* **Biological Inspiration:** The journey started in the 1950s with the Perceptron, a rigid artificial neuron with a hard yes/no threshold.
* **Modern Neurons:** To make models trainable, modern deep learning uses smooth, continuous neurons (using activation functions like Sigmoid or ReLU) that can output probabilities rather than just binary answers.
* **Network Depth:** Architectures evolved from "shallow" networks (just one hidden layer) to "deep" networks (multiple consecutive hidden layers) capable of hierarchical feature extraction.

## 2. How the Models Actually Learn
* **Gradient Descent:** The process of mathematically stepping down a "loss landscape" to find the point of minimum error.
* **Batch vs. Stochastic (SGD):** While Batch Gradient Descent processes the whole dataset for a smooth path, SGD looks at single samples (or mini-batches) for dramatically faster, noisier training that helps escape bad local minima.
* **Backpropagation:** The core learning engine. It propagates the error backward through the layers to figure out exactly how much each weight needs to be adjusted.
* **The Adam Optimizer:** A modern upgrade to standard SGD that uses "momentum" to power through tricky areas and adaptively changes the learning rate for different parameters.

## 3. Improving Accuracy & Robustness
* **Weight Decay (L2 Regularization):** Penalizes massively large weights to prevent the model from memorizing noise (overfitting).
* **Dropout:** Randomly turns off a percentage of neurons during training to force the network to build redundant, robust pathways rather than relying on a single feature.
* **Weight Initialization:** Starting the network off with specifically scaled weights to prevent gradients from vanishing or exploding.

## 4. Specialized Architectures
* **Vision (CNNs):** Convolutional Neural Networks shift away from flat layers to use spatial filters. They scan across 2D images to extract everything from basic edges to complex shapes for object detection.
* **Sequences (RNNs):** Recurrent Neural Networks are built for time-series data (like text or audio). They update continuously, keeping a running "snowball" summary of the past to predict the future.
* **Memory (LSTMs):** Because standard RNNs suffer from "catastrophic forgetting" on long sequences, LSTMs introduce an uninterrupted "cell state" conveyor belt and specialized gates to decide exactly what to remember and what to forget over long periods. Solve long term dependency problem.