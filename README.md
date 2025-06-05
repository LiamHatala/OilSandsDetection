# Oil Sands Level Detection 

### 1. Training the model
Make sure you are in the main directory - DeepLabV3Plus-Pytorch and then run the following command
```python
python3 -m train.train
```
### 2. Hyperparameter tuning 
Make sure you are in the main directory - DeepLabV3Plus-Pytorch and then run the following command
```python
python3 -m train.tuning
```

### 3. Visualization of the results
Make sure you are in the main directory - DeepLabV3Plus-Pytorch and then run the following command
Change the value of the variable video_pth to the path of the video to be visualized
```python
python3 -m utils.visualizer 
```
