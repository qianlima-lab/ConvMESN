# Convolutional Multi-timescale Echo State Network
The code in this repository for paper "Convolutional Multi-timescale Echo State Network" accepted by IEEE Transactions on Cybernetics.



### Dependencies

* Keras 2.0 and above
* tensorflow 0.4 and above



### 18 MTS Datasets

The 18 MTS(Multivariate Time Series) benchmark data sets can download from [link](https://pan.baidu.com/s/1lLP36LkngMedROSjDSM9qg). The verification code is `b29j`. They are collected from different repository, such as UCI, UCR and so on. These benchmark data sets come from various fields and have various input ranges and different numbers of classes, variables, and instances. Detail information of each data set is shown in following Fig. 

And the folder `datasets` contain a demo dataset `ECG` .  The file `ECG.p`  is a list of three numpy arrays with `[samples, lables, original_lengths]`.  Meanwhile,

```
samples.shape = (number of instances, time length, number of variables)
labels.shape = (number of instances,)
original_lengths.shape = (number of instances,)
```

![Fig](F:\github-lab\ConvMESN\MTS.JPG)



### Usage

You can run the command `python ConvMESN.py` to test the model ConvMESN.







