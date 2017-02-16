# Nature Conservancy Fish Classification
Scripts/notebooks for The Nature Conservancy's fish classification competition

- Current best score before blending: 0.886 log loss on LB from `ensemble_model.ipynb` using vgg16bn architecture.
- Current best score after blending: 0.822 log loss on LB from blending `ensemble_model.ipynb` results with `conv_model.ipynb` results

### ensemble_model.ipynb
Fits a group of CNNs of (optionally) varying architectures on the image data; predicts on augmented test data

### conv_model.ipynb
Computes and saves the convolutional features from a pretrained VGG model w/ batch normalization; uses these features to train a new model on the image data

### pytorch_model.ipynb
Uses pretrained PyTorch models with both train-time and test-time augmentation. Have only tested resnet architecture so far, but currently produces third best single model (after "ensemble_model" and "conv_model"). 

### fish_detector.ipynb
Trains a model to distinguish images that contain fish from those that don't

### end_to_end.ipynb
Similar to `ensemble model`, but uses the "fish detector" model to segment out the "NoF" class first

### bb_crops.ipynb
Uses annotations to crop training images down to a bounding box surrounding the fish

### bb_splits.ipynb
Util notebook for splitting cropped data into training and test sets

### bb_regressor.ipynb
Trains model on cropped data to predict coordinates of bounding box in test images

### bb_end_to_end.ipynb
Similar to `end_to_end` but uses the "bb regressor" model to try to crop test images first

### bb_multi_input.ipynb
Mutli-input CNN using keras functional API to incorporate bounding box coordinates as a feature in the model; predicts both coordinates and class probabibilities

### tf_svm.ipynb
Uses conv features of image data as input to SVM model

### tf_xgb.ipynb
Same as above, but using gradient boosting instead of SVM

### sliding_window.ipynb
Uses a sliding window to feed subsets of image into CNN for classification

### submission_blender.ipynb
Combines submission files into simple ensemble

## Other Notes:
- Unable to get any improvement over vanilla CNNs using any of the bounding box, cropping, sliding window, or pre-filtering strategies above. In particular, the `bb_end_to_end` model scored significantly worse -- my theories include: 
  -  not enough accuracy from bb regressor to make the stategy work
  - size/aspect ratio mismatches introduced at some point in the pipeline
  - user error in the pipeline somewhere (ie filenames not aligned with predictions, etc)
- Relabeling the dataset also didn't lead to improved performance using these methods
- Unable to get comparable performance from Resnet and Inception models. Might just be an issue of my implementation

(credit to <a href="http://course.fast.ai/">Jeremy Howard</a>, <a href="https://github.com/pengpaiSH/Kaggle_NCFM">Pai Peng</a>, <a href="https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/25902"> Naive Shuai</a>, <a href="https://www.kaggle.com/craigglastonbury/the-nature-conservancy-fisheries-monitoring/using-inceptionv3-features-svm-classifier">Craig Glastonbury</a>, and others for portions of this code)
