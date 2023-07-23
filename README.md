# sparse-RNNs

Code implementations in Python for sparsifying recurrent neural networks (RNNs) using pruning or rewiring strategies, along with measuring the structural balance of the sparsified recurrent neural networks. These codes were used for the experiments described in the paper: X.-J. Zhang, et al. Universal structural patterns in sparse recurrent neural networks.

## Table of Contents
- [Background](#background)
- [Requirements](#requirements)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Background
The purpose of this project is to provide Python implementations of pruning and rewiring strategies for sparsification of recurrent neural networks. By using these algorithms, you can reduce the number of connections (trainable parameters) in an RNN while maintaining its performance. Additionally, the code includes functionality to measure the structural balance of the sparsified recurrent neural network, as described in the aforementioned research paper.

## Requirements
If the environment configurations are different, the results may vary greatly or even fail to work properly. You may need to adjust the code details according to your needs.

- Python 3.6.0
- tensorflow 1.15.0
- tensorboard 1.15.0
- numpy 1.19.5
- networkx 2.5.1
- seaborn 0.11.2
- scipy 1.5.4

## Usage
To use this code, import the necessary modules and functions into your Python script. You can then call the provided methods to implement sparsification using pruning and rewiring, measure the structural balance, and perform other related tasks.

## Examples
The project includes examples with several datasets, such as MNIST, SMS, Gestrue, and Mackey_Glass. Additionally, it provides examples with different network architectures, including RNNs, neural ordinary differential equation networks (Neural ODEs), and continuous-time RNNs (CT-RNNs), as mentioned in the paper.

There is a separate folder for each dataset:
- MNIST: MNIST_pruning.py, MNIST_rewiring.py, MNIST_trainer.py
- SMS: SMS_pruning.py, SMS_rewiring.py, SMS_trainer.py
- Mackey-Glass: MG_pruning.py, MG_rewiring.py, MG_trainer.py
- Gesture: Gesture_pruning.py, Gesture_rewiring.py, Gesture_trainer.py

The experimental analysis in the paper is contained in the analysis folder, including:
- Measuring structural balance: calculate_triad_balanced.py, triad_balance.py
- Motif lesion experiments: lesion_mnist.py, lesion_mg.py, lesion_gesture.py

## License
This project is licensed under the [MIT License](LICENSE).