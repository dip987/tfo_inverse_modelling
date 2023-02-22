# Normalization
|Normalization Method | Training Error | Validation Error| Comments |
----------------------|----------------|-----------------|----------|
|Everything between 0 to 1 | 0.015 (log avg) | 0.019 (log avg)| SGD, lr 0.001|
|Zero mean(between -0.5 to +0.5) | Similar | Similar | Seems a lot more stable|
|Zero mean(between -1 to + 1)  | Lower, 0.011| Similar | Validation loss was higher at the beginning|
  
