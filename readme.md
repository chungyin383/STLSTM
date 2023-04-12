# Skeleton-Based Action Recognition Using Spatio-Temporal LSTM Network with Trust Gates

This is an implementation of the above paper, which is available at https://arxiv.org/pdf/1706.08276v1.pdf. We implement Leave-One-Out-Cross-Validation (LOOCV) on the UTKinect dataset.

## Requirements

`pip install pytorch numpy tqdm`

## How to use

1. Download UTKinect dataset, which can obtained [here](https://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html). 

2. Run `main.py` and optionally specifiying arguments, e.g. `python main.py --learning_rate 2e-3`. The whole list of available arguments are shown below.

3. `result.csv` will be created to record the loss and accuracy on training and validation respectively.

## Arguments
- `num_sub_seq`: Number of sub-sequences, default value = 10
- `dataset_root`: Path of your dataset root, default value = './data/UTKinect'
- `num_epochs`: Number of epochs, default value = 1000
- `batch_size`: Batch size, default value = 256
- `input_size`: Input size, default value = 3
- `num_layers`: Number of ST-LSTM layers, default value = 2
- `hidden_size`: Size of hidden state, default value = 32
- `with_trust_gate`: Whether to use the trust gate mechanism introduced in the paper. You can input 'Y' or 'N', 'Y' means with trust gate, 'N' means otherwise. Default value = 'Y'.
- `tree_traversal`: Whether to use the Tree Traversal algorithm specified in the paper. You can input 'Y' or 'N', 'Y' means with tree traversal, 'N' means ordinary joints order. Default value = 'Y'.
- `learning_rate`: learning rate, default value = 1e-2
- `end_factor`: end_factor of linear scheduler, default value = 1e-2
- `total_iters`: total_iters of linear scheduler, default value = 100
