# 3D Hand Gestures Recognition Using Pointnet

**Using a PointNet neural network and MediaPipe to recognize Sign Language symbols (an other hand gestures) in 3D**

## How to use it

### 1. Install the requirements

```bash
pip install -r requirements.txt
```

### 2. Run the script

```bash
python3 . -h
```

## How it works

### 1. Data collection

To collect the data, we use the MediaPipe library to detect the hand landmarks and the OpenCV library to capture the video. The data is saved in a .json file located in the `data/raw` folder. To run the harvest data script, use the following command:

```bash
python3 src/webcam_harvest.py -h
```

### 2. Data preprocessing

The data preprocessing is done in the `src/preprocess.ipynb` notebook. It is used to clean the data and to create the training and test sets. The data is saved in a .json file located in the `data/clean` folder. To run the preprocessing script, run the `src/preprocess.ipynb` notebook.

### 3. Training

The training is done in the `src/train.ipynb` notebook. It is used to train the PointNet neural network. The model is saved in a .pth file located in the `data/model` folder. To run the training script, run the `src/train.ipynb` notebook.

## Donations

If you want to support me, you can donate me on [PayPal](https://paypal.me/danimelchor)
