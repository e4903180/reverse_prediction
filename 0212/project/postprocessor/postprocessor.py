import numpy as np
import pandas as pd

class Postprocesser:
    def __init__(self):
        pass
    
    def check_shape(self, X_train, X_test, X_newest, y_train, y_test, y_preds, reshape="False"):
        if reshape == "True":
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])
            X_newest = X_newest.reshape(X_newest.shape[0], X_newest.shape[2], X_newest.shape[1])
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[2], y_train.shape[1])
            y_test = y_test.reshape(y_test.shape[0], y_test.shape[2], y_test.shape[1])
            y_preds = y_preds.reshape(y_preds.shape[0], y_preds.shape[2], y_preds.shape[1])
        return X_train, X_test, X_newest, y_train, y_test, y_preds
    
    def modify_rows(self, arr):
        for row in arr:
            # Find the first index in the row where the change occurs
            change_index = np.where(np.diff(row) != 0)[0]
            if change_index.size > 0:
                first_change_index = change_index[0] + 1
                # Set all values after the first change to the value at the change index
                row[first_change_index:] = row[first_change_index]
        return arr

    def remove_short_sequences(self, arr, x):
        """
        Remove sequences in the array that are shorter than x, considering both 0 to 1 and 1 to 0 changes.

        :param arr: The input array
        :param x: The minimum sequence length to keep
        :return: The modified array
        """
        # Identify the changes in the array
        change_indices = np.where(np.diff(arr) != 0)[0] + 1
        # Include the start and end of the array
        change_indices = np.insert(change_indices, 0, 0)
        change_indices = np.append(change_indices, len(arr))
        
        for i in range(len(change_indices) - 1):
            # Calculate the length of the sequence
            seq_length = change_indices[i+1] - change_indices[i]
            if seq_length < x:
                # Set the values of short sequences to the value preceding the sequence
                arr[change_indices[i]:change_indices[i+1]] = arr[change_indices[i] - 1]
        return arr

    def process_signals(self, y_data, dates, filter):
        max_indices = np.argmax(y_data, axis=-1)
        # max_indices = self.modify_rows(max_indices)
        flatten_max_indices = max_indices.flatten()
        if filter != 'False':
            flatten_max_indices = self.remove_short_sequences(flatten_max_indices, filter)
        signals = np.full(flatten_max_indices.shape, '', dtype=object)

        for i in range(1, len(flatten_max_indices)):
            # downward to upward
            if flatten_max_indices[i-1] == 1 and flatten_max_indices[i] == 0:
                signals[i] = 'Buy'
            # upward to downward
            elif flatten_max_indices[i-1] == 0 and flatten_max_indices[i] == 1:
                signals[i] = 'Sell'

        non_empty_signals = np.where(signals != '')[0]
        if non_empty_signals.size > 0:
            first_signal_index = non_empty_signals[0]
            last_signal_index = non_empty_signals[-1]
            signals[first_signal_index] += ' (first)'
            signals[last_signal_index] += ' (last)'

        flat_dates = dates.flatten()
        return pd.DataFrame({'Date': flat_dates, 'Signal': signals})
