# Training DenseNet on Quickdraw

### Preprocessing quickdraw data

* Download quickdraw data
```
    git clone https://github.com/googlecreativelab/quickdraw-dataset.git
```

* Download the raw images of the quickdraw-dataset
```
    python download_npy.py
```

* Generate the train/val/test split by giving the number of training samples from each category
```
    python split_data.py
```

### Train densenet on the generated training split

* Train the 100-layer densenet using PyTorch (make sure the PyTorch is correctly installed)
```
    cd densenet.pytorch
    python train_quickdraw.py
```

* Test the trained model. The script is [here](densenet.pytorch/test_instance.py).
```
    python test_instance.py
```
