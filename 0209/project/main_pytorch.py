
import json
from preprocessor.preprocessor_pytorch import Preprocessor
from model.model_pytorch import Model
from postprocessor.postprocessor import Postprocesser
from evaluator.evaluator_pytorch import Evaluator
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import torch


def set_seed(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)


def main():
    with open('parameters.json', 'r') as file:
        params = json.load(file)

    preprocessor = Preprocessor()
    data = preprocessor.fetch_stock_data(
        params['stock_symbol'], params['start_date'], params['stop_date'])
    for single_feature_params in params['features_params']:
        feature_type = single_feature_params["type"]
        data = preprocessor.add_feature(
            data, feature_type, **single_feature_params)
    data, issues_detected = preprocessor.add_data_cleaner(data,
                                                          clean_type=params['data_cleaning']['clean_type'], 
                                                          strategy=params['data_cleaning']['strategy'])
    X_train, y_train, X_test, y_test, train_dates, test_dates = \
        preprocessor.process_data(data, split_ratio=params['split_ratio'], target_col=params['target_col'],
                                  feature_cols=None, look_back=params['look_back'],
                                  predict_steps=params['predict_steps'],
                                  train_slide_steps=params['train_slide_steps'],
                                  test_slide_steps=params['test_slide_steps'])
    X_newest, x_newest_date = preprocessor.create_x_newest_data(
        data, params['look_back'])

    model_wrapper = Model()
    model, history, y_preds, online_training_losses, online_training_acc = \
        model_wrapper.run(params['model_type'], params['look_back'], params['model_params'][params['model_type']],
                          X_train, y_train, X_test, y_test)

    postprocessor = Postprocesser()
    test_trade_signals = postprocessor.process_signals(y_test, test_dates)
    pred_trade_signals = postprocessor.process_signals(y_preds, test_dates)

    evaluator = Evaluator()
    evaluator.generate_report(model, y_test, y_preds, history,
                              online_training_acc, online_training_losses,
                              data, pred_trade_signals, x_start=0, x_stop=300)


if __name__ == "__main__":
    set_seed(42)
    main()
