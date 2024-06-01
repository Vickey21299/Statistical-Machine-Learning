# Waste Classifier for Degradable and Non-Biodegradable Wastes
[View Full Report ]([https://github.com/Vickey21299/Statistical-Machine-Learning/blob/main/SML_PROJECT/SML_PROJECT/final_ppt_2021121_2021299.pdf](https://github.com/Vickey21299/Statistical-Machine-Learning/blob/main/SML_PROJECT/SML_PROJECT/final_report_2021299_2021121.pdf))
## Team Members
- Abhishek IIITD (2021121)
- Vickey Kumar (2021299)

## Abstract
Inefficient waste sorting poses a significant challenge to environmental sustainability. Traditional methods, relying on manual sorting of degradable and non-biodegradable waste, are slow and prone to errors. We propose developing an automated waste classifier utilizing Convolutional Neural Networks (CNNs) to address this. This project aims to streamline and improve waste sorting processes by analyzing image data of waste objects, enhancing environmental sustainability.

## Problem Being Addressed
Effective waste management is crucial for environmental sustainability. Manual sorting of waste is time-consuming and error-prone. Automated waste classifiers streamline sorting, ensuring accuracy and efficiency. Development of such a classifier aims to improve waste management and environmental conservation.

## Literature Review
Reduction techniques such as PCA and LDA for feature extraction and classification may not capture intricate image details as effectively as CNNs. CNNs are chosen for their proven effectiveness in image classification and their ability to learn complex features. The decision to use CNNs is based on their superior performance in handling image data and learning intricate visual characteristics.

## Dataset
- **Source:** [Kaggle Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- **Features:** Images of waste items
- **Labels:**
  - Degradable (1)
  - Non-Biodegradable (0)
- **Train Folder Composition:**
  - Organic (O)
  - Recyclable (R)
- **Total Images:** 22,564

## Methodology
1. **Data Collection:** Gather a diverse dataset of waste images.
2. **Data Preprocessing:** Clean and standardize the images.
3. **Model Selection:** Choose a pre-trained CNN model.
4. **Fine-tuning:** Adapt the model to waste classification.
5. **Training:** Train the model on the preprocessed dataset.
6. **Evaluation:** Assess model performance using metrics.
7. **Optimization:** Refine model parameters and architecture.
8. **Deployment:** Implement the model for automated waste classification.
9. **Testing and Validation:** Verify system performance with new data.
10. **Feedback Incorporation:** Continuously improve the system based on user feedback.

## Models Tried and Results
- **LDA:**
  - Accuracy: 32%
  - Class 0: 18%
  - Class 1: 44%
- **Decision Trees:**
  - Accuracy: 44%
  - Class 0: 11%
  - Class 1: 70%
- **CNN:**
  - Accuracy: 88%
  - Class 0: 63%
  - Class 1: 37%

## Datasets Used
- Created new datasets using data augmentation techniques like flipping around y-axis, random rotation, flipping around x-axis, and adding noise.

## Experimental Settings, Results, Comparisons
How we reached the proposed architecture:
- Researched existing image classification projects, finding CNNs widely used and effective.
- Experimented with simpler architectures but found them inadequate for capturing complex waste features.

Dataset Preprocessing:
- Grayscale Conversion: Converted images to grayscale for simplicity and reduced computational complexity.
- Resizing: Standardized all images to 64x64 pixels to facilitate machine learning.

## Motivation
Why we need to use CNN:
- CNNs excel in image processing due to automatic hierarchical feature learning, translation invariance, parameter efficiency, and handling complexity of image data.
- CNNs preserve spatial relationships in images, making them superior to LDA and Decision Trees for image processing tasks.

## Advanced Version of CNN: VGG16 Model
- VGG16 is a 16-layer deep neural network with a total of 138 million parameters.
- The simplicity of the VGGNet16 architecture is its main attraction.
- The VGGNet architecture incorporates the most important convolution neural network features.

## Output
Desirable output for loss and accuracy for validation data and train data:
- Preliminary Accuracy using Decision Tree and AdaBoost: 45%
- Final Accuracy using VGG16: 89%

## Conclusion
In summary, CNNs emerge as the preferred choice for automated waste classification due to their inherent capabilities in feature learning, robustness to image variations, parameter efficiency, and performance on large datasets. Their ability to handle image complexity and preserve spatial relationships makes them indispensable for accurate and efficient waste management practices.

---

[View Full Report (final_ppt_2021121_2021299.pdf)](https://github.com/Vickey21299/Statistical-Machine-Learning/blob/main/SML_PROJECT/SML_PROJECT/final_ppt_2021121_2021299.pdf)
