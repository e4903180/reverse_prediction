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
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import time
import pickle


def set_seed(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)


def main(current_dir=os.getcwd()): 
    with open(os.path.join(current_dir, 'parameters.json'), 'r') as file:
        params = json.load(file)
        
    os.makedirs(os.path.join(current_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'data'), exist_ok=True)

    preprocessor = Preprocessor(params)
    X_train, y_train, X_val, y_val, X_test, y_test, train_dates, test_dates, X_newest, x_newest_date, SP500 = preprocessor.get_multiple_data()
    
    start_time = time.time()
    model_wrapper = Model(params=params)
    model, history, y_preds, online_training_losses, online_training_acc = \
        model_wrapper.run(X_train, y_train, X_test, y_test, X_val, y_val, current_dir)
    end_time = time.time()
    execution_time = end_time - start_time
    
    y_preds = torch.tensor(y_preds, dtype=torch.float32)
    y_preds = preprocessor.change_values_after_first_reverse_point(y_preds)

    postprocessor = Postprocesser()
    X_train, X_test, X_newest, y_train, y_test, y_preds = \
        postprocessor.check_shape(X_train, X_test, X_newest, y_train, y_test, y_preds, reshape=params['model_params'][params['model_type']]['reshape'])
    test_trade_signals = postprocessor.process_signals(y_test, test_dates, False)
    pred_trade_signals = postprocessor.process_signals(y_preds, test_dates, params['filter'])

    evaluator = Evaluator(params=params)
    evaluator.generate_report(model, y_test, y_preds, history, 
                            online_training_acc, online_training_losses,
                            SP500, pred_trade_signals, test_trade_signals, execution_time, current_dir, x_start=0, x_stop=300)

    file_path = os.path.join(current_dir, 'data\\data.pkl')
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'X_newest': X_newest,
        'y_train': y_train,
        'y_test': y_test,
        'y_preds': y_preds
    }
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    open('progress.txt', 'w').close()
    open('log.txt', 'w').close()
    root_path = 'projects'
    for floder in tqdm.tqdm(os.listdir(root_path), file=open('progress.txt', 'a')):
        first_path = os.path.join(root_path, floder)
        for subfloder in tqdm.tqdm(os.listdir(first_path), file=open('progress.txt', 'a')):
            second_path = os.path.join(first_path, subfloder)
            print(second_path, file=open('progress.txt', 'a'))
            set_seed(42)
            try: 
                main(second_path)
                print('done', file=open('progress.txt', 'a'))
            except Exception as e:
                print(e, file=open('progress.txt', 'a'))
                continue
            
