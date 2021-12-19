# Training Protocol

## Model

The main segmentation model is an EfficientUnet++ with a EfficientNet-b5 encoder.

## Losses

The model was trained using a combination of three losses and a loss weighting scheme.  
![L = (1 - \alpha) * (L_{gd} + L_{fo}) + (\alpha * L_{bd})](https://latex.codecogs.com/svg.latex?L%20%3D%20%28%281%20-%20%5Calpha%29%20*%20%28L_%7Bgd%7D%20&plus;%20L_%7Bfo%7D%29%29%20&plus;%20%28%5Calpha%20*%20L_%7Bbd%7D%29)

and ɑ (ramped on over the first 100 epochs) is defined as:  
![\alpha = min(0.01*epoch, 0.99)](https://latex.codecogs.com/svg.latex?%5Calpha%20%3D%20min%280.01*epoch%2C%200.99%29)

### Generaliced Dice Loss
...
### Boundary loss
...
### Focal Loss
...

## Training

The following `pytorch lightning` settings and training tricks were used:  
- [x] `batch_size`: 32
- [x] `ADAM optimizer` with `learning_rate` of 0.0003 (altered using CosineAnnealingLR with T_max=10) 
- [x] mixed precision training 
- [x] gradient clipping enabled (mode: norm, 0.5)
- [x] stochastic weight averaging  

## Data

- [x] Data samples of size 4x256x256 (CxWxH; channels: R,G,B,NIR), normalized  
- [x] Data augmentation:  
      - `HorizontalFlip` or `VerticalFlip`, _p=0.5_  
      - `RandomRotate90`, _p=0.5_  
      - `RandomBrightnessContrast`, _brightness_limit=0.2_, _contrast_limit=0.15_, _brightness_by_max=False_  
      - `Normalize`  
- [x] Normalization: 4-channel mean/ std for all data from 2017-2020
