import numpy as np
import pandas as pd


class Postprocesser:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def process_signals(self, y_data, dates):
        max_indices = np.argmax(y_data, axis=-1)
        flatten_max_indices = max_indices.flatten()
        signals = np.full(flatten_max_indices.shape, '', dtype=object)

        for i in range(1, len(flatten_max_indices)):
            # downward to upward
            if flatten_max_indices[i-1] == 1 and flatten_max_indices[i] == 0:
                signals[i] = 'Buy'
            # upward to downward
            elif flatten_max_indices[i-1] == 0 and flatten_max_indices[i] == 1:
                signals[i] = 'Sell'

        flat_dates = dates.flatten()
        return pd.DataFrame({'Date': flat_dates, 'Signal': signals})
