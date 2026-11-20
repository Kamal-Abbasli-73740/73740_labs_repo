\# Project Proposal: Offline Tennis Shot Classification on CPU



\*\*Course:\*\* Machine Learning Systems (CII5SE039CI) – Group 02  

\*\*Student:\*\* Kamal Abbasli (ID: 73740)  



\---



\## 1. Problem and Goal

When playing tennis, it is useful to track how many forehands, backhands, and serves you hit during practice. Usually, recognizing shots from photos or video requires sending images to an online server, which needs a constant internet connection on the court and creates delay. My goal is to build a small image classification model that recognizes tennis shots directly on a normal CPU offline, without needing internet or expensive hardware.



\## 2. Dataset and Preprocessing

I will use a public Tennis Shots image dataset from Kaggle with four classes: \*\*Forehand\*\*, \*\*Backhand\*\*, \*\*Serve\*\*, and \*\*Ready/Idle\*\*. I will split the data into 70% for training, 15% for validation, and 15% for testing. 



Before feeding the photos into the model, I will resize all images to 128x128 pixels and normalize the color values. During training, I will apply basic data augmentation like horizontal flips and small brightness changes so the model does not overfit and handles different court lighting better.



\## 3. Baseline Model

For the starting baseline, I will build a lightweight Convolutional Neural Network (CNN) in PyTorch (comparing a simple 4-block custom CNN and a small MobileNetV2). I will train it using Cross-Entropy Loss and the Adam optimizer, and log all training runs and metrics in MLflow. This baseline will give me the starting numbers for accuracy, file size, and CPU speed before I compress the model.



\## 4. Success Metrics and Constraints

To make sure the model is practical on a low-power device, it needs to meet these targets:

\* \*\*Accuracy:\*\* At least 85% accuracy on the test set (and around 0.84 Macro F1-score).

\* \*\*File Size:\*\* The final saved model must be \*\*5 MB or smaller\*\*.

\* \*\*Speed:\*\* Predicting a single image (batch size 1) on a CPU must take \*\*50 ms or less\*\*.



\## 5. Optimization and Compression Plan

To hit the 5 MB and 50 ms limits, I will optimize the baseline model in three steps:

1\. \*\*Structured Pruning:\*\* Remove about 25% of the less useful filters in the convolutional layers and fine-tune the model for a few epochs so the CPU has fewer calculations to run.

2\. \*\*INT8 Quantization:\*\* Convert the model weights and activations from 32-bit floats to 8-bit integers using PyTorch static quantization. This should cut the file size by about 4 times and speed up CPU inference.

3\. \*\*Model Export:\*\* Export the compressed model to TorchScript and ONNX Runtime to remove extra Python overhead when running predictions.



\## 6. Evaluation Plan

I will test and compare four versions of the model on the exact same CPU:

1\. Original FP32 baseline model

2\. Pruned FP32 model

3\. INT8 quantized model

4\. Pruned + INT8 exported model (TorchScript / ONNX)



For each version, I will measure the test accuracy, Macro F1-score, file size in MB, peak memory usage, and average single-image CPU inference time in milliseconds.



\## 7. Potential Risks and Backup Plans

\* \*\*Small dataset or messy court backgrounds:\*\* If the dataset is too small or background colors confuse the model, I will use stronger image augmentation and rely on pretrained weights from a scaled-down MobileNetV2 so it generalizes better.

\* \*\*Accuracy drop after compression:\*\* Pruning and quantization can lower accuracy. To fix this, I will retrain the model briefly after pruning and calibrate the INT8 model using validation images so accuracy stays above 85%.

