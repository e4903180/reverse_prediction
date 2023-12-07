from preprocessor.preprocessor import Preprocessor
from model.model import Model
from postprocessor.postprocessor import Postprocesser
from evaluator.evaluator import Evaluator
import numpy as np
import tensorflow as tf
import random


def set_seed(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)


def main():
    model_data = Preprocessor()
    model_data.set_seed(42)
    start_date = "2001-01-01"
    stop_date = "2021-01-01"
    stock_symbol = "^GSPC"
    model_data.data = model_data.fetch_stock_data(
        stock_symbol, start_date, stop_date)

    features_params = [
        {"type": "Trend", "method": "MA", "oder_days": 20,
         "ma_days": 20, "trend_days": 5},
        {"type": "MACD", "fastperiod": 5, "slowperiod": 10, "signalperiod": 9},
        {"type": "ROC", "trend_days": 5},
        {"type": "Stochastic Oscillator", "trend_days": 5},
        {"type": "CCI", "timeperiod": 14},
        {"type": "RSI", "timeperiod": 14},
        {"type": "VMA", "timeperiod": 20},
        {"type": "PctChange"},
        {"type": "3M Treasury Yield", "start_date": "2001-01-01",
            "end_date": "2021-01-01"},
        {"type": "5Y Treasury Yield", "start_date": "2001-01-01",
            "end_date": "2021-01-01"},
        {"type": "10Y Treasury Yield",
            "start_date": "2001-01-01", "end_date": "2021-01-01"},
        {"type": "30Y Treasury Yield",
            "start_date": "2001-01-01", "end_date": "2021-01-01"},
    ]  # Add other features here as needed

    for single_feature_params in features_params:
        feature_type = single_feature_params["type"]
        model_data.add_feature(feature_type, **single_feature_params)

    issues_detected = model_data.add_data_cleaner(
        "MissingData", strategy='auto')
    split_ratio = 0.7
    target_col = "Trend"
    feature_cols = None  # None means use all columns
    # feature_cols = ['Trend']
    look_back = 64  # number of previous days' data to consider
    predict_steps = 16  # number of days to predict in the future
    train_slide_steps = 1  # sliding window step size

    X_train, y_train, X_test, y_test, train_dates, test_dates = \
        model_data.process_data(split_ratio=split_ratio, target_col=target_col, 
                                feature_cols=feature_cols, look_back=look_back,
                                predict_steps=predict_steps, 
                                train_slide_steps=train_slide_steps, 
                                test_slide_steps=predict_steps)
    X_newest, x_newest_date = model_data.create_x_newest_data(look_back)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(X_newest.shape)

    params = {
        'conv_1_filter': 32,
        'conv_1_kernel': 4,
        'conv_1_l2': 0.00016475815766673417,
        'dropout_1': 0.2,
        'conv_2_filter': 96,
        'conv_2_kernel': 2,
        'dropout_2': 0.2,
        'lstm_1_units': 128,
        'lstm_1_l2': 0.0002788818914602332,
        'dropout_3': 0.2,
        'lstm_2_units': 64,
        'dropout_4': 0.1,
        'learning_rate': 0.001,
        'look_back': look_back,
        'predict_steps': predict_steps
    }
    pre_trained_model_path = 'model_20231117_144448.h5'
    model_type = 'seq2seq'
    model_wrapper = Model()
    # TODO: add a method to use more parameters
    model, history, y_preds, online_training_losses, online_training_acc = \
        model_wrapper.run(model_type, look_back, params,
                          X_train, y_train, X_test, y_test)
    # model, history, y_preds, online_training_losses, online_training_acc = \
    #     model_wrapper.run(model_type, look_back, params, X_train, y_train, 
    #                       X_test, y_test, pre_trained_model_path=pre_trained_model_path)

    postprocessor = Postprocesser()
    test_trade_signals = postprocessor.process_signals(y_test, test_dates)
    pred_trade_signals = postprocessor.process_signals(y_preds, test_dates)
    # newest_trade_signals = postprocessor.process_signals(y_newest, y) # TODO: fix this
    evaluator = Evaluator()
    evaluator.analyze_results(y_test, y_preds, history,
                              online_training_acc, online_training_losses)
    backtest_results = evaluator.perform_backtesting(
        model_data.data, pred_trade_signals)
    for strategy in backtest_results:
        print("Final Portfolio Value: ", strategy.broker.getvalue())
    print("Sharpe Ratio: ", strategy.analyzers.sharpe_ratio.get_analysis())
    print("Drawdown Info: ", strategy.analyzers.drawdown.get_analysis())
    print("Trade Analysis: ", strategy.analyzers.trade_analyzer.get_analysis())


if __name__ == '__main__':
    set_seed(42)
    main()
