## Object-based CNN (OCNN) for satellite imagery semantic labeling 
 
The goal of OCNN is to provide a fast, accurate way for the semantic labelling of satellite images while keeping detail information about geographical entities. It is designed to be easy implement to support satellite imagery mapping and evaluations of benchmark research. If you find this to be helpful, please cite our works 

[“Object-based convolutional neural network for high-resolution imagery classification, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 10(7), 3386-3396”](https://ieeexplore.ieee.org/abstract/document/7890382/)

["Spectral–spatial feature extraction for hyperspectral image classification: A dimension reduction and deep learning approach,IEEE Transactions on Geoscience and Remote Sensing 54 (8), 4544-4554”](https://ieeexplore.ieee.org/abstract/document/7450160/)

In addition, we would thank **Prof. Thomas Blaschke, Prof. Stefan Lang, Prof. Dirk Tiede and the OBIA group members for their valuable suggestions**. 

**Note:**

The Object-based CNN (OCNN) has already integrated Per-pixel CNN (PCNN) strategy, so we waived the Matlab version of PCNN since its efficiency seemed a bit of low.

To use the OCNN codes, you may want to make sure the necessary environment already satisfied. The related modules or packages are:

- Tensorflow
- cv2
- pickle

Other basic modules such as numpy, scipy, PIL also should be installed.

**The overall structure** (It’s the prototype, so maybe looks a bit of crumble): 

```
|-OCNN_main.py (improtant!)
|-save_samples.py (important!)
|-sample_extraction
|-CNN_models.py
|-graph.py
|-smooth_filter.py
|-samples_data
|- - training/test samples (in pickle files)
```

Commonly, if you wanna train a model from scratch and then predict satellite imagery based on it, there are 3 steps you may  follow.

### Step. 1: Generate training dataset using “save_samples.py” 
In this file, you should assign certain parameters before dataset generation. For instance:
tr_number : proportion of training samples 
te_number: proportion of test samples
patchshape: sample sizes
img_path: image directory
gt_path: reference map directory (labels should begin from 1, and 0 as the background)

Then, run this file, training and test samples can be automatically generated under the directory of “samples_data”

### Step. 2: Train the OCNN model using “OCNN_main.py”
In this file, you should set the parameters from line 22 to 35 as standard CNN configuration. Then, switch to Line 170 “FLAGS.train = 1(means training) and 2(means predicting)”.
Then, run this file for standard CNN training. 

### Step. 3: Predict the satellite imagery labels using well-trained CNN model.

Just back to the Line 170 “OCNN_main” and set to “FLAGS.train = 2” for image prediction. Keep in mind, the predicted results may be presented in the format of pixel- (Line 226) or object-based (Line 247) results. 

Please feel free to modify our OCNN model to fit in your researches and purposes. If you have any questions, please send me an e-mail. Wish you can enjoy it! 
