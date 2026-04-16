# CNN-Based-COVID-19-Diagnostic-Support-System-Multi-Task-Transfer-Learning-on-Chest-X-Ray-Images
A deep learning-based diagnostic support system for chest X-ray analysis using DenseNet and multi-task learning. The model also performs multiclass classification (COVID-19, Lung Opacity, Viral Pneumonia, Normal), lung segmentation, and explainability via Grad-CAM, achieving 92% accuracy and high Dice/IoU scores

This project presents a deep learning–based diagnostic support system for chest X-ray analysis using multi-task learning and transfer learning.

The system integrates three core capabilities that is
1. Multiclass classification of chest X-ray images
2. Pixel-level lung segmentation
3. Explainability using Grad-CAM

The model classifies chest X-rays into four clinically relevant categories.

COVID-19
Lung Opacity
Viral Pneumonia
Normal

Unlike traditional binary classification approaches, this system reflects real-world clinical scenarios, where multiple respiratory diseases must be distinguished simultaneously. This improves diagnostic relevance and practical usability in healthcare settings.

# Objectives

The key objectives of this project are

1. Develop a robust multiclass CNN model for chest X-ray diagnosis
2. Evaluate multi-task learning vs single-task CNN baselines
3. Improve interpretability using Grad-CAM visualizations
4. Integrate lung segmentation for anatomically guided learning
5. Enhance diagnostic performance through transfer learning strategies

# Dataset
Source: Kaggle COVID-19 Radiography Database and link https:https:https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
Total Images: 21,165 chest X-rays with corresponding lung masks
Dataset Includes: Chest X-ray images, Pixel-level lung segmentation masks

# Class Distribution of the dataset
Class	Count
Normal	10,192
Lung Opacity	6,012
COVID-19	3,616
Viral Pneumonia	1,345 (Note: The dataset is imbalanced, with the Normal class dominating and Viral Pneumonia underrepresented)

# Data Preparation
Training: 70%
Validation: 15%
Test: 15%

Preprocessing Steps included Resizing all images to 224 × 224 pixels, Normalize pixel values to [0,1]
converting to RGB format, applying data augmentation (horizontal flips, small rotations) and Remove corrupted/duplicate images.Stratified splitting was used to preserve class distribution across datasets.

# Methodology
1. Baseline Models
The following pretrained CNN architectures were evaluated
EfficientNet
ResNet
DenseNet
Observation during this stage - EfficientNet and ResNet showed poor performance ( around 48% accuracy), while DenseNet achieved significantly better results.

2. Model Selection
DenseNet was selected as the backbone due to
Strong feature reuse (dense connectivity)
Superior classification performance (82% baseline accuracy)

3. Fine-Tuning Strategy
Unfroze the last 40 layers
Reduced learning rate to 1e-5
Improved performance to 89% accuracy

4. Multi-Task Learning Framework
The proposed architecture consists of
Shared DenseNet encoder
Classification head (disease prediction)
Segmentation head (lung mask prediction)

This allows the model to learn shared representations that improve both tasks.

5. Weight Transfer (Key Innovation)
Instead of using ImageNet weights directly. The multi-task model was initialized using fine-tuned DenseNet weights. This provided domain-specific feature representations,Faster convergence and Improved stability and performance. This is a key contribution of the project.

# Model Architecture
🔹 Encoder
DenseNet121 (pretrained and fine-tuned)
🔹 Classification Head
Global Average Pooling
Dense Layer (512 units)
Dropout (0.3)
Softmax Output (4 classes)
🔹 Segmentation Head
U-Net–style convolutional decoder
Skip connections
Sigmoid activation (binary lung mask)

# Results
Final Model Performance (Multi-Task Phase 1.5)
Metric	Value
Accuracy	0.92
AUC	0.986
F1 Score	0.91
Dice Score	0.89
IoU	0.80

# Baseline Comparison
Model	Accuracy
EfficientNet	0.48
ResNet	0.48
DenseNet	0.82
Fine-Tuned DenseNet	0.89

(Note: DenseNet significantly outperformed other architectures in both accuracy and F1-score)

# Explainability (Grad-CAM)
Grad-CAM was applied to highlight important regions influencing predictions,validate model focus on lung regions and improve clinical interpretability

The approach extracts gradients from the final convolutional layer and overlays heatmaps on input images to visualize model attention.

# Segmentation Insights
Dice Score: 0.89
IoU Score: 0.80

Note: Segmentation improves performance by guiding the model toward lung regions, reducing false activations outside anatomical areas. Enhancing classification robustness.

# Link for the Project Video Recording

# Contributors
Peter Mvuma pmvuma@mtu.edu/mvumapeter@gmail.com
Mohammed Yushawu Abdulai myabdula@mtu.edu/amyshhgh@gmail.com
