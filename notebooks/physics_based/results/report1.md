
Applying Physics Loss
=====================

# Objective


Using 80% Random Split to predict mu_a difference with the help of pathlength as a physics-loss term  

# Data Length


75600  

# Model Used


```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
PerceptronBD                             --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       1,025
│    └─BatchNorm1d: 2-2                  50
│    └─ReLU: 2-3                         --
│    └─Linear: 2-4                       650
│    └─BatchNorm1d: 2-5                  50
│    └─ReLU: 2-6                         --
│    └─Linear: 2-7                       650
│    └─BatchNorm1d: 2-8                  50
│    └─ReLU: 2-9                         --
│    └─Linear: 2-10                      650
│    └─BatchNorm1d: 2-11                 50
│    └─ReLU: 2-12                        --
│    └─Linear: 2-13                      650
│    └─BatchNorm1d: 2-14                 50
│    └─ReLU: 2-15                        --
│    └─Linear: 2-16                      650
│    └─BatchNorm1d: 2-17                 50
│    └─ReLU: 2-18                        --
│    └─Linear: 2-19                      572
│    └─Flatten: 2-20                     --
=================================================================
Total params: 5,147
Trainable params: 5,147
Non-trainable params: 0
=================================================================
```  

# Loss


Label Loss(training): 0.06971703387844135,
                       Label loss(validation): 0.06421293703322652,
                       Physics Loss(training): 1.7274199539042263e-05
                       Physics Loss(validation): 1.71339461609145e-05  

# Loss Functions Used


Label Loss & BL Pathlength Loss, with weights: [1.0, 10000.0]  

# Model Trainer Params


```

        Model Properties:
        PerceptronBD(
  (model): Sequential(
    (0): Linear(in_features=40, out_features=25, bias=True)
    (1): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=25, out_features=25, bias=True)
    (4): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Linear(in_features=25, out_features=25, bias=True)
    (7): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
    (9): Linear(in_features=25, out_features=25, bias=True)
    (10): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU()
    (12): Linear(in_features=25, out_features=25, bias=True)
    (13): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): Linear(in_features=25, out_features=25, bias=True)
    (16): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (17): ReLU()
    (18): Linear(in_features=25, out_features=22, bias=True)
    (19): Flatten(start_dim=1, end_dim=-1)
  )
)
        Optimizer Properties"
        SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    lr: 0.001
    maximize: False
    momentum: 0.91
    nesterov: False
    weight_decay: 0.0001
)
        DataLoader Params: 
            Batch Size: 4096
            Validation Method: Split the data randomly using np.random.shuffle with a split of 0.8
        Loss:
            Train Loss: 0.24245902971535857
            Val. Loss: 0.23555239822183335
```  

# Loss Curves
  
  
![Loss Curves](figures/report1_6.png)  

# Prediction & Error Distribution
  
  
![Prediction & Error Distribution](figures/report1_7.png)  
