# Handwritten Digit Recognition: Neural Network from Scratch

> A small neural network that classifies handwritten digits from the Kaggle Digit Recognizer dataset, written in NumPy with no TensorFlow, PyTorch or Keras.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

**[Kaggle Notebook →](https://www.kaggle.com/code/tobioparinde/practice-neural-network-digit-recognition)**

## Overview

This project trains a neural network to recognise handwritten digits (0–9) from 28 × 28 pixel images. I implemented the network in NumPy, following a tutorial, including forward propagation, backpropagation and gradient descent, so I could see how each step works instead of calling a framework. Pandas loads the CSV and Matplotlib displays the images.

After training, I added an error analysis of my own to see which digits the model finds hardest and which wrong predictions it makes with high confidence.

On a held-out dev set of 1,000 images, the trained model reached **86.20% accuracy**.

---

## What I Built

**Architecture**

| Layer | Size | Activation |
|---|---|---|
| Input | 784 (28 × 28 pixels) | n/a |
| Hidden | 10 neurons | ReLU |
| Output | 10 classes (digits 0–9) | Softmax |

**Implemented from scratch**

- **Parameter initialisation**: random weights and biases for both layers
- **ReLU**: hidden-layer activation
- **Numerically stable softmax**: subtracts the column maximum before exponentiating to avoid overflow
- **One-hot encoding**: turns digit labels into target vectors for the loss gradient
- **Forward propagation**: computes the class probabilities for a batch of images
- **Backpropagation**: computes gradients for all weights and biases
- **Gradient descent**: updates the parameters using those gradients

---

## Dataset and Approach

The data is the labelled `train.csv` from the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition. Each row holds a digit label followed by 784 pixel values.

- **X** is the input: the 784 pixel values of each image, scaled to 0–1, with one image per column.
- **Y** is the target: the true digit (0–9) for each image.

I shuffled the labelled data and held out **1,000 images as a dev (validation) set**. The rest were used for training. The dev set is never used to update the weights, so it gives a fairer read on unseen images than training accuracy does.

**Training settings**

- Iterations: 500
- Learning rate: 0.1
- Optimiser: full-batch gradient descent

---

## Results and Error Analysis

Final accuracy on the 1,000-image dev set: **86.20%**.

A single overall number hides where the model is weak, so I also calculated accuracy for each digit:

| Digit | Dev images | Accuracy |
|---|---|---|
| 5 (hardest) | 90 | 73.33% |
| 8 | 84 | 77.38% |
| 9 | 115 | 80.00% |
| 4 | 88 | 84.09% |
| 7 | 100 | 86.00% |
| 3 | 110 | 87.27% |
| 2 | 89 | 87.64% |
| 6 | 102 | 91.18% |
| 0 | 106 | 91.51% |
| 1 (strongest) | 116 | 99.14% |

**Error analysis (my own extension).** The basic network only needed an accuracy score, but I wanted to know where it fails. I displayed the five incorrect dev predictions with the highest predicted probability, meaning the cases where the model was most confident and still wrong. The images are shown in the notebook.

| Predicted | Actual | Probability |
|---|---|---|
| 0 | 9 | 99.24% |
| 6 | 5 | 98.76% |
| 9 | 8 | 95.34% |
| 7 | 9 | 93.59% |
| 7 | 9 | 91.98% |

Looking at the images, both of the 9s predicted as 7 have a small loop at the top with a slanted tail, and the 5 predicted as 6 has a rounded lower curve. These are quick observations from five images, not a systematic analysis. Three of the five errors involve a true 9, and 9 is one of the three lowest-scoring digits, but five examples are too few to draw firm conclusions.

Checking mistakes was useful because per-digit accuracy showed which classes to look at first, and the confident errors showed the model can be wrong with high certainty, so its probabilities shouldn't be read as reliability.

> **Note:** Results may change between runs. The data is shuffled and the parameters are randomly initialised without a fixed seed. The figures above come from one saved run and were measured on my dev set only, not on Kaggle's separate test set, so they are not a competition score.

---

## How to Run

The notebook is designed for Kaggle and reads its data from:

```
/kaggle/input/competitions/digit-recognizer/train.csv
```

1. Create a Kaggle notebook and import this one.
2. Attach the **Digit Recognizer** competition data.
3. Run the notebook from top to bottom.

**Libraries:** NumPy, Pandas, Matplotlib

---

## What I Learned and Next Steps

**What I learned**

- How forward propagation, backpropagation and gradient descent connect, by implementing each one by hand
- Why softmax needs to be numerically stable, and how one-hot labels feed into the output-layer gradient
- Why a held-out set matters, and how per-digit accuracy and confident mistakes tell me more than one overall score

**Next steps**

- Compare different hidden-layer sizes to see how much the 10-neuron hidden layer limits accuracy
- Set a random seed so results are reproducible between runs
- Look more closely at the digits the model finds hardest (5, 8 and 9)

---

## Acknowledgements

Based on [Samson Zhang’s neural network math tutorial](https://www.youtube.com/watch?v=w8yWXqWQYmU).

The per-digit accuracy and confident-mistake analysis is my own extension.

---

*Built with Python, NumPy, Pandas and Matplotlib.*
