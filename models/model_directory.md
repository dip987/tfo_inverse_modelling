## Remarks
1. The x and y scalers for each model are stored as model_name_xscaler and model_name_yscaler. Use them to scale the data before applying the model.
2. For building the features passed into the model, use the the functions inside __features/build_features.py__


File Name | Model Object | Instantiation Params |  Fitting | Remark
----------|--------------|----------------------|----------|-------
fsat_delta_5det_v1| PerceptronReLU | [20, 10, 4, 2, 1] | 1.0, 0.8 | Trained on simulation to predict Fetal Saturation Delta. Intended to transfer to Real life data. Uses both IR and SI as logs
fsat_percep_irsi|PerceptronReLU | [60, 8, 1] | 1.0, 0.8 | Trained on simulation for fetal saturation
