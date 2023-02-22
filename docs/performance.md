# Normalization
|Normalization Method | Training Error | Validation Error| Comments |
|---------------------|----------------|-----------------|----------|
|Everything between 0 to 1 | 0.015 (log avg) | 0.019 (log avg)| SGD, lr 0.001|
|Zero mean(between -0.5 to +0.5) | Similar | Similar | Seems a lot more stable|
|Zero mean(between -1 to + 1)  | Lower, 0.011| Similar | Validation loss was higher at the beginning|
  
# Predicting Both Wavelengths
Requires a bit longer training to get results. Using SGD

|Network | Training Error | Validation Error| Epochs |LR       | Comments |
|--------|----------------|-----------------|--------|---------|----------|
|[5, 4, 3, 2] | 0.015     | 0.010           | 25     | 0.002   |Takes a few epochs to get a low Val error. Should go even lower with higher epochs|
|[5, 5, 4, 3, 2] | 0.022  | 0.02            | 25     | 0.005   | Should also be going down|

# Predicting All 20 points

