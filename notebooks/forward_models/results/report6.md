
Forward Modelling Report (WV1)
==============================

# Objective


4 Layer network with weight decay in SGD  

# Data Length


48384  

# Model Used


```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
PerceptronBD                             --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       30
│    └─BatchNorm1d: 2-2                  10
│    └─Dropout1d: 2-3                    --
│    └─ReLU: 2-4                         --
│    └─Linear: 2-5                       60
│    └─BatchNorm1d: 2-6                  20
│    └─Dropout1d: 2-7                    --
│    └─ReLU: 2-8                         --
│    └─Linear: 2-9                       165
│    └─BatchNorm1d: 2-10                 30
│    └─Dropout1d: 2-11                   --
│    └─ReLU: 2-12                        --
│    └─Linear: 2-13                      320
│    └─Flatten: 2-14                     --
=================================================================
Total params: 635
Trainable params: 635
Non-trainable params: 0
=================================================================
```  

# Model Trainer Params


```

        Model Properties:
        PerceptronBD(
  (model): Sequential(
    (0): Linear(in_features=5, out_features=5, bias=True)
    (1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Dropout1d(p=0.1, inplace=False)
    (3): ReLU()
    (4): Linear(in_features=5, out_features=10, bias=True)
    (5): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Dropout1d(p=0.1, inplace=False)
    (7): ReLU()
    (8): Linear(in_features=10, out_features=15, bias=True)
    (9): BatchNorm1d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): Dropout1d(p=0.1, inplace=False)
    (11): ReLU()
    (12): Linear(in_features=15, out_features=20, bias=True)
    (13): Flatten(start_dim=1, end_dim=-1)
  )
)
        Optimizer Properties"
        SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    lr: 0.0003
    maximize: False
    momentum: 0.89
    nesterov: False
    weight_decay: 0.0001
)
        DataLoader Params: 
            Batch Size: 32
            Validation Method: Holds out fMaternal Wall Thickness columns -0.6546536707079772 for validation. The rest are used             for training
        Loss:
            Train Loss: 0.002059768156876375
            Val. Loss: 0.01304747973859468
```  

# Loss Curves
  
  
![Loss Curves](figures/report6_4.png)  

# Prediction Distribution
  
  
![Prediction Distribution](figures/report6_5.png)  

# Error Distribution
  
  
![Error Distribution](figures/report6_6.png)  
