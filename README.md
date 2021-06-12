# Image_Classification_ResNet

Multi-label Image Classification with ResNet

Dataset used: Large-scale CelebFaces Attributes (CelebA) Dataset
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Feature Visualization Techniques:
(1) Principal component analysis (PCA)
(2) t-SNE
(3) Guided backpropagation
(4) Grad-CAM

The directory structure of the project is as follows:
───input
│   └───face-classifier
|       ├───Images
|       |   │   Images from CelebA dataset
│       │   │
│       └───Multi_Label_dataset
│           │   identity_CelebA.txt
│           │   list_attr_celeba.txt
|           |   list_bbox_celeba.txt
|           |   list_eval_partition.txt
|           |   list_landmarks_align_celeba.txt
|           |   list_landmarks_celeba.txt
│           
│                   
├───outputs
│
├───src
|   │   dataset.py
|   │   engine.py
|   │   inference.py
|   │   models.py
|   │   train.py
|
└───Visualization
    │   grad-CAM.py
    |   guide_backward.py
    |   PCA.py
    |   t_SNE_visualization.py
