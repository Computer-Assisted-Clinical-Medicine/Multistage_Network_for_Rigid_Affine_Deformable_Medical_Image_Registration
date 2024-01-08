# Multistage Network for Rigid Affine Deformable Medical Image Registration
This repository contains the implementation of a multistage network for rigid-affine-deformable medical image registration.
In order to provide a benchmark for comparison, we integrated a multistage network for affine-deformable registration into the framework. 
The framework was implemented using Keras with Tensorflow backend.

If you use our code in your work please cite the following paper:

Strittmatter A, Caroli A, Zöllner FG. A Multistage Rigid-Affine-Deformable Network for Three-Dimensional Multimodal Medical Image Registration. Applied Sciences. 2023; 13(24):13298. https://doi.org/10.3390/app132413298 

# Architecture
![Architecture_github](https://github.com/Computer-Assisted-Clinical-Medicine/Multistage_Network_for_Rigid_Affine_Deformable_Medical_Image_Registration/assets/129390849/20de7e14-087d-4e9a-9e39-ddd293ed8b78)

# Manual
Usage:
1. In config.py:
    
   a) Provide the file paths to your data: fixed images, moving images, and segmentations (if available) (lines 10-13).
   
   b) Specify the path where the results should be saved (line 14) and the name of the output folder (line 17).
   
   c) If needed, also add the path to the pre-trained weights (line 15).
   
   d) Provide the paths to the CSV files containing the filenames of the fixed images, moving images, and segmentations (line 22, 27, 32, 37).
   
   e) If segmentations are available, set seg_available = True (line 62).
   
4. Run the main program (main.py) with the following settings:
   
   a) To train a network from scratch: set is_training = True (line 426). After training, the weights for each fold of the five-fold cross-validation will be saved in subfolders within the output folder (numbered from 0 to 4).
   
   b) To finetune a network: set is_pretrained = True (line 427). In config.py, specify the path to the pre-trained weights (line 15). The folder should contain separate subfolders for each fold of the five-fold cross-validation (numbered from 0 to 4). These subfolders should contain the pre-trained weights for each fold in a file named "weights.h5". After finetuning, the weights for each fold of the five-fold cross-validation will be stored in subfolders within the output folder (numbered from 0 to 4).
   
   c) For interference and evaluation, set is_apply = True and is_evaluate = True (line 428 and 429). The output folder will contain two subfolders: "predict" and "eval". The "predict" folder will contain the registered moved images and segmentations (if seg_available = True in config.py line 62). The "eval" folder will contain the Dice coefficient values (if seg_available = True in config.py line 62) and number of Jacobian determinants ≤ 0 for each registered image pair. Additionally, boxplots for the metrics and individual folds of the five-fold cross-validation will be stored in "eval/plots".
   
   d) The network architecture can be selected in lines 442-449. The architectures can be found in NetworkBasis.networks.py. 

