![build](https://github.com/fratajcz/pytorch_hyperbolic/actions/workflows/build.yml/badge.svg)
![pypi](https://github.com/fratajcz/pytorch_hyperbolic/actions/workflows/pypi.yml/badge.svg)

# pytorch_hyperbolic
Implementation of hyperbolic NNs and GNNs 

# Installation

```
pip install torch-hyperbolic
pip install torch-sparse
```

or 

```
git clone https://github.com/fratajcz/pytorch_hyperbolic.git
cd pytorch_hyperbolic
pip install .
pip install -r requirements.yml
```

# Usage

The individual layers can be found in ```torch_hyperbolic.nn```. For an example how they can be used, in this case for node classification, can be found in ```torch_hyperbolic.models.hgnn```. The implementation is kept very simple so that the reader can quickly identify what is going on and adapt it to his or her needs. 

A very bare-bone example of a training script can be found in ```example_train.py```, which trains a node classification task on the Cora dataset. After installation, it can be started as follows:

```
$python example_train.py

$ python example_train.py 
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.x
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.tx
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.allx
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.y
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ty
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ally
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.graph
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.test.index
    Processing...
    Done!
    Model curvatures before training: [1.0, 1.0, 1.0, 1.0]
    Epoch: 4, Train Loss: 0.6340, Val Loss: 0.6468
    Epoch: 9, Train Loss: 0.5058, Val Loss: 0.5066
    Epoch: 14, Train Loss: 0.4296, Val Loss: 0.4553
    Epoch: 19, Train Loss: 0.3863, Val Loss: 0.4190
    Epoch: 24, Train Loss: 0.3640, Val Loss: 0.4007
    Epoch: 29, Train Loss: 0.3450, Val Loss: 0.3895
    Epoch: 34, Train Loss: 0.3284, Val Loss: 0.3765
    Epoch: 39, Train Loss: 0.3149, Val Loss: 0.3665
    Epoch: 44, Train Loss: 0.3012, Val Loss: 0.3593
    Epoch: 49, Train Loss: 0.2873, Val Loss: 0.3550
    Epoch: 54, Train Loss: 0.2735, Val Loss: 0.3456
    Epoch: 59, Train Loss: 0.2594, Val Loss: 0.3378
    Epoch: 64, Train Loss: 0.2446, Val Loss: 0.3276
    Epoch: 69, Train Loss: 0.2286, Val Loss: 0.3177
    Epoch: 74, Train Loss: 0.2120, Val Loss: 0.3116
    Epoch: 79, Train Loss: 0.1948, Val Loss: 0.3043
    Epoch: 84, Train Loss: 0.1775, Val Loss: 0.2970
    Epoch: 89, Train Loss: 0.1609, Val Loss: 0.2919
    Epoch: 94, Train Loss: 0.1450, Val Loss: 0.2886
    Epoch: 99, Train Loss: 0.1300, Val Loss: 0.2874
    Test Loss: 0.28
    Model curvatures after training: [1.136, 0.977, 0.318, 0.172]

```

As you can see, the data is downloaded and processed. Then, the model is initialized with trainable curvatures at a starting value of 1. After training for 100 Epochs, the training stops and gives us the loss on the test set.
Finally, we can see that the learned curvatures have changed. A low value close to 0 indicates a low hyperbolicity, i.e. a behaviour of the layers comparable to their euclidean counterpart, while a high curvature indicates a high hyperbolicity.
