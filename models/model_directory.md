## Remarks
1. The x and y scalers for each model are stored as model_name_xscaler and model_name_yscaler. Use them to scale the data before applying the model.
2. For building the features passed into the model, use the the functions inside __features/build_features.py__


File Name | Model Object | Instantiation Params |  Fitting | Remark
----------|--------------|----------------------|----------|-------
fsat_delta_5det_v1| PerceptronReLU | [20, 10, 4, 2, 1] | 1.0, 0.8 | Trained on simulation to predict Fetal Saturation Delta. Intended to transfer to Real life data. Uses both IR and SI as logs
fsat_percep_irsi|PerceptronReLU | [60, 8, 1] | 1.0, 0.8 | Trained on simulation for fetal saturation
forward_curve_fit_paramv1 | PerceptronBD | [5, 8, 8, 8], 'dropout_rates': [0.2, 0.2] | 1.0, 0.8 | A forward model that takes in the 5 TMPs and predicts 8 fitting parameters alpha0_1, alpha0_2, alpha1_1, alpha1_2, ... (Look into forward_model_s_based.ipynb)
forward_curve_fit_paramv1 | PerceptronBD | [5, 4, 8, 16, 40], 'dropout_rates': [0.01, 0.01, 0.01]} | A forward model that takes in 5 TMPs and spits out the spatial log10(intensity) at wv1 followed by wv2, pretty solid performance!
