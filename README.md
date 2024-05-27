LightGBMModelOptimizer Python-package
=======================

Usage
*****
LightGBMModelOptimizer optimized the trained lightgbm model. It reduces size of model & improves the inference time of the model.

To Optimize the trained model without dumping. It will create a new model object and return it.

    optimizer = Optimizer()
    model = optimizer.optimize_booster(model)

To Optimize the model dump. It will replace the same file with the optimized model file.

    optimizer = Optimizer()
    _ = optimizer.optimize_model_file('model.txt')


