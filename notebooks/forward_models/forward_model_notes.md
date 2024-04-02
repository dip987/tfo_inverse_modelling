# Training the Forward Model
It seems that a deeper model larger than 1 kernel length works better (in terms of training). The prediction distribution seems a bit side-heavy. Used tanh to try to counter this effect. But the non-tanh'd version works better.   
If you look at the error distribution, some detectors in-between have worse validation scores. Whereas, I would expect the farther we go, the worse the errors become. Which is absolutely not true!  
Test out batch sizes: Better results at 32. 