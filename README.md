# cv_final_assignment  <sub>- Park JaeYun (2021111633)<sub>

## Summary of the Final Approach
This project uses ***AlexNet pretrained on ImageNet*** as the base model.
#### Overall Accuracy: ***86.43%*** (1306 / 1511 correct predictions)

I tested several strategies, including extended data augmentation, different optimization methods (Adam, AdamW), and various training-configuration changes.

Among them, the final approach achieved 86% test accuracy. For more details on the most effective setup, please refer to the Training Pipeline section and the code below.

The best-performing ***.pth file*** is available through the provided download link.
https://drive.google.com/file/d/14urhJuVp43QSBOSefFlxbNh2DIwa1wf9/view?usp=drive_link

## Training Pipeline
The training pipeline consists of several stages, ranging from dataset preparation to model optimization and evaluation.
Below is the complete step-by-step workflow used for fine-tuning AlexNet on the pathology image dataset.

### 1. Dataset Preparation
1. All images are stored in class-specific directories:
```text
Training/
 ├── Chorionic_villi/
 ├── Decidual_tissue/
 ├── Hemorrhage/
 └── Trophoblastic_tissue/
```
2. Each image path is collected and stored in a DataFrame.
3. Class names are encoded into integer labels using LabelEncoder.

### 2. Train/Validation Split
- The dataset is split into 70% Training and 30% Validation.
- A stratified split ensures all classes maintain their original distribution in both sets.

### 3. Data Transformation & Augmentation
Two separate transformation pipelines are used:
1. Training Transformations
- Resize → CenterCrop
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation (±30°)
- ColorJitter (brightness, contrast, saturation, hue)
- Convert to Tensor
- Normalize

2. Validation Transformations
- Resize → CenterCrop
- Convert to Tensor
- Normalize

### 4. Data Transformation & Augmentation
1. A custom PyTorch Dataset class:
- Loads images using PIL
- Applies transformations
- Returns (image_tensor, label)

2. DataLoaders:
- Batch Size: 32
- Training: shuffle = True
- Validation: shuffle = False
- num_workers = 2 for multi-processing loading

### 5. Data Transformation & Augmentation
1. Load AlexNet pretrained on ImageNet
2. Replace the last fully connected layer (1000 → 4)
3. Move model to GPU if available (cuda:0)

### 6. Optimization Setup
1. Optimizer & Loss
- Optimizer: Adam (learning rate = 0.0001)
- Loss Function: CrossEntropyLoss

2. Learning Rate Scheduler
- StepLR(step_size = 7, gamma = 0.1)
  → Reduces the learning rate by a factor of 0.1 every 7 epochs.

3. Batch & Epoch Settings
- Batch Size: 32
- Epochs: 10
- Train/Validation Split: 70% / 30% (stratified)

### 7. Training Loop
For each epoch, the model is trained and validated as follows:
1. Training Phase
- Set model to train()
- Forward pass → compute loss
- Backward pass → update weights (optimizer.step())
- Track loss and accuracy (displayed with tqdm)

2. Validation Phase
- Set model to eval()
- Disable gradients (torch.no_grad())
- Forward pass → compute validation loss & accuracy

3. Model Checkpointing
- Save the model weights whenever validation accuracy improves.

### 8. Model Saving
After training, the model is saved as a .pth file using the weights that achieved the best validation accuracy.
- best validation accuracy: 93.53% (Epoch 8)

## Test Pipeline
The trained AlexNet-based classifier was evaluated on an independent test dataset across the same four tissue classes. 
The testing procedure followed the same preprocessing and label-encoding pipeline used during training to ensure consistency.

### 1. Test Data Preparation
- Images were loaded from the Testing/ directory.
```text
Testing/
 ├── Chorionic_villi/
 ├── Decidual_tissue/
 ├── Hemorrhage/
 └── Trophoblastic_tissue/
```
- The same LabelEncoder mapping used during training was applied to preserve label consistency.
- Validation transforms (Resize → CenterCrop → Normalize) were used to avoid augmentation during testing.

### 2. Test Dataloader
- Batch Size: 32
- No shuffling (to preserve sample order)
- num_workers: 2

### 3. Inference Procedure
For all test samples:
1. Model set to eval() mode
2. torch.no_grad() for efficient inference
3. Forward pass
4. Predicted class determined via argmax
5. Predictions and ground-truth labels stored

### 4. Metrics Computed
- Accuracy
- Precision, Recall, F1-score (per-class)
- Classification report using scikit-learn

## Test Results
#### Overall Accuracy: 86.43% (1306 / 1511 correct predictions)
### Per-Class Performance
<img width="876" height="554" alt="image" src="https://github.com/user-attachments/assets/ee6447d9-8337-4e2e-99d7-f50f483f36c9" />

### Result Interpretation
1. Strong performance on most classes
- Chorionic_villi and Hemorrhage classes show high recall (0.98 and 0.96)
- Trophoblastic_tissue shows high precision (0.96)
  
2. Weaker performance on Decidual_tissue
- Recall = 0.63 → The model often misclassifies this class as others.

3. Balanced macro & weighted averages (0.86)
- Model has consistent performance across classes, not dominated by high-support categories.
