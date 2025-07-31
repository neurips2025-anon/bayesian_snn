# SUMMARY

This repository contains the training code for the experiments reported in  
**"A Principled Bayesian Framework for Training Binary and Spiking Neural Networks"**.

For each dataset:

- To implement our Bayesian Binary/Spiking Neural Networks, use:  
  `--method "bayesian" --scale_style "learnable"`

- To implement the surrogate gradient method, use:  
  `--method "sg" --scale_style "fixed"`

Each experiment includes a training script and preprocessed data. 

---

# REQUIREMENTS

```
pip install -r requirements.txt
```

---

# CIFAR-10 Experiment

### Files
- `train_cifar.py`: Main training script  
- The required data will be automatically downloaded and stored in `~/data` when the training script is exectuted. 

### Run Training

```
python train_cifar.py --method "bayesian" --scale_style "learnable" --batch_size 256 --epochs 200 --time_steps 1 --seed 3
```

### Arguments

- `--method`: Training method (`"bayesian"` or `"sg"`)  
- `--scale_style`: Controls whether the posterior variance is learned or fixed (`"learnable"` or `"fixed"`)  
- `--batch_size`: Batch size (default: 256)  
- `--epochs`: Number of training epochs (default: 200)  
- `--time_steps`: Number of time steps (1 for binary networks, >1 for spiking networks)  
- `--seed`: Random seed for reproducibility

### Output

Model checkpoint is saved as:  
`{method}_cifar10.pth`

---

# DVS Gesture Experiment

### Files

- `train_dvs.py`: Main training script  
- `dvs_data.py`: Dataset definition and preprocessing for DVS Gesture chains  
- Required data provided in the folder `~/DVSGC_frames_number_49_split_by_number`

### Run Training

```
python train_dvs.py --method "bayesian" --scale_style "learnable" --batch_size 16 --epochs 70 --time_steps 49
```

### Arguments

- `--method`: Training method (`"bayesian"` or `"sg"`)  
- `--scale_style`: Controls whether the posterior variance is learned or fixed (`"learnable"` or `"fixed"`)  
- `--batch_size`: Training batch size  
- `--epochs`: Number of training epochs  
- `--time_steps`: Number of time steps (1 for binary networks, >1 for spiking networks)  
- `--seed`: Random seed for reproducibility

### Output

Trained model saved as:  
`{method}_dvs.pth`

---

# SHD Experiment

### Files

- `train_shd.py`: Main training script  
- `shd_data.py`: Dataset download, preprocessing, and caching  
- The required data will be automatically downloaded and stored in `~/hdspikes` when the training script is exectuted. 

### Run Training

```
python train_shd.py --method "bayesian" --scale_style "learnable" --batch_size 32 --epochs 30 --time_steps 100
```

### Arguments

- `--method`: Training method (`"bayesian"` or `"sg"`)  
- `--scale_style`: Controls whether the posterior variance is learned or fixed (`"learnable"` or `"fixed"`)  
- `--batch_size`: Batch size for training  
- `--epochs`: Number of epochs  
- `--time_steps`: Number of time steps (1 for binary networks, >1 for spiking networks)  
- `--seed`: Random seed for reproducibility

### Output

Model checkpoint is saved as:  
`{method}_shd.pth`
