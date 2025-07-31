========
SUMMARY
========
This repository contains the training code for the experiments reported in "A Principled Bayesian Framework for Training Binary and Spiking Neural Networks". 

For each dataset, to implement our Bayesian Binary/Spiking Neural networks use the following:
--method "bayesian" --scale_style "learnable" 

To implement the surrogate gradient method, use:
--method "sg" --scale_style "fixed" 

Each experiment includes a training script and preprocessed data (already included in the repository). No additional downloads are required.


========
REQUIREMENTS
========
pip install -r requirements.txt


========
CIFAR-10 Experiment
========

--------
Files
--------
- train_cifar.py: Main training script  
- The required data is stored in ~/data

--------
Run Training 
--------
    python train_cifar.py --method "bayesian" --scale_style "learnable" --batch_size 256 --epochs 200 --time_steps 1 --seed 3


--------
Arguments  
--------
--method        : Training method ("bayesian" or "sg")  
--scale_style   : Scale initialization ("learnable" or "fixed")  
--batch_size    : Batch size (default: 256)  
--epochs        : Number of training epochs (default: 200)  
--time_steps    : Number of time steps (default: 1)  
--seed          : Random seed for reproducibility

--------
Output  
--------
Saves model checkpoint as:  
    {method}_cifar10.pth


========
DVS Gesture Experiment
========

--------
Files
--------
- train_dvs.py: Main training script  
- dvs_data.py: Dataset definition and preprocessing for DVS Gesture chains
- The required data is stored in ~/DVSGC_frames_number_49_split_by_number

--------
Running the script
--------
    python train_dvs.py --method "bayesian" --scale_style "learnable" --batch_size 16 --epochs 70 --time_steps 49

--------
Arguments
--------
--method        : Training method ("bayesian" or "sg")  
--scale_style   : Noise scaling mode ("learnable" or "fixed")  
--batch_size    : Training batch size  
--epochs        : Number of training epochs  
--time_steps    : Temporal window size (e.g., 49 for full gesture)  
--seed          : Random seed for reproducibility

--------
Output
--------
Trained model saved as:  
    {method}_dvs.pth



========
SHD Experiment
========

--------
Files
--------
- train_shd.py: Main training script  
- shd_data.py: Dataset download, preprocessing, and caching
- The required data is stored in ~/hdspikes


--------
Running the script
--------
    python train_shd.py --method "bayesian" --scale_style "learnable" --batch_size 32 --epochs 30 --time_steps 100


--------
Arguments
--------
--method        : Training method ("bayesian" or "sg")  
--scale_style   : Noise scaling method ("learnable" or "fixed")  
--batch_size    : Batch size for training  
--epochs        : Number of epochs  
--time_steps    : Length of the temporal window (default: 100)  
--seed          : Random seed for reproducibility


--------
Output
--------
Model checkpoint is saved as:  
    {method}_shd.pth
