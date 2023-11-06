
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import talib
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Indicator(ABC):
    """
    Abstract base class for all indicators.
    """
    @abstractmethod
    def compute(self, data=None, *args, **kwargs):
        """
        Abstract method to compute the indicator value for the given data.
        """
        pass


# TODO: there's some wrong in calculate trend ways, add a new method that can split data by date
class TrendIndicator(Indicator):
    """
    Indicator to calculate the trend based on various methods.
    """
    def compute(self, data, *args, **kwargs):
        """
        Compute the trend for the given data using the specified method.

        Parameters:
        - data: DataFrame containing the data.
        - method: Method for trend calculation (e.g., 'MA', 'LocalExtrema').
        - ma_days: Number of days for moving average.
        - oder_days: Number of days for order.
        - trend_days: Number of days to determine the trend.

        Returns:
        - DataFrame with trend values.
        """
        method = kwargs.get('method', 'MA')
        ma_days = kwargs.get('ma_days', 20)
        oder_days = kwargs.get('oder_days', 20)
        trend_days = kwargs.get('trend_days', 5)

        if method == 'MA':
            return self.calculate_trend_MA(data, ma_days=ma_days, trend_days=trend_days)
        elif method == 'LocalExtrema':
            return self.calculate_trend_LocalExtrema(data, oder_days=oder_days)
        else:
            raise ValueError(f"Invalid trend calculation method: {method}")

    def calculate_trend_MA(self, data, ma_days=20, trend_days=5):
        """
        Calculate trend using Moving Average method.

        Parameters:
        - data: DataFrame containing the data.
        - ma_days: Number of days for moving average.
        - trend_days: Number of days to determine the trend.

        Returns:
        - DataFrame with trend values.
        """
        data['MA'] = data['Close'].rolling(window=ma_days).mean()
        data['Trend'] = np.nan
        n = len(data)

        for i in range(n - trend_days + 1):
            if all(data['MA'].iloc[i + j] < data['MA'].iloc[i + j + 1] for j in range(trend_days - 1)):
                data['Trend'].iloc[i:i + trend_days] = 0
            elif all(data['MA'].iloc[i + j] > data['MA'].iloc[i + j + 1] for j in range(trend_days - 1)):
                data['Trend'].iloc[i:i + trend_days] = 1
        data['Trend'].fillna(method='ffill', inplace=True)
        return data.drop(columns=['MA'])

    def calculate_trend_LocalExtrema(self, data, oder_days=20):
        """
        Calculate trend using Local Extrema method.

        Parameters:
        - data: DataFrame containing the data.
        - oder_days: Number of days for order.

        Returns:
        - DataFrame with trend values.
        """
        local_max_indices = argrelextrema(
            data['Close'].values, np.greater_equal, order=oder_days)[0]
        local_min_indices = argrelextrema(
            data['Close'].values, np.less_equal, order=oder_days)[0]
        data['Local Max'] = data.iloc[local_max_indices]['Close']
        data['Local Min'] = data.iloc[local_min_indices]['Close']
        data['Trend'] = np.nan
        prev_idx = None
        prev_trend = None
        prev_type = None

        for idx in sorted(np.concatenate([local_max_indices, local_min_indices])):
            if idx in local_max_indices:
                current_type = "max"
            else:
                current_type = "min"

            if prev_trend is None:
                if current_type == "max":
                    prev_trend = 1
                else:
                    prev_trend = 0
            else:
                if prev_type == "max" and current_type == "min":
                    data.loc[prev_idx:idx, 'Trend'] = 1
                    prev_trend = 1
                elif prev_type == "min" and current_type == "max":
                    data.loc[prev_idx:idx, 'Trend'] = 0
                    prev_trend = 0
                else:
                    if current_type == "max":
                        data.loc[prev_idx:idx, 'Trend'] = 0
                        prev_trend = 0
                    else:
                        data.loc[prev_idx:idx, 'Trend'] = 1
                        prev_trend = 1

            prev_idx = idx
            prev_type = current_type
        data['Trend'].fillna(method='ffill', inplace=True)
        return data.drop(columns=['Local Max', 'Local Min'])


class MACDIndicator(Indicator):
    """
    Indicator to calculate the Moving Average Convergence Divergence (MACD).
    """

    def compute(self, data, *args, **kwargs):
        fastperiod = kwargs.get('fastperiod', 5)
        slowperiod = kwargs.get('slowperiod', 10)
        signalperiod = kwargs.get('signalperiod', 9)
        data['MACD'], _, _ = talib.MACD(
            data['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return data


class ROCIndicator(Indicator):
    def compute(self, data, *args, **kwargs):
        trend_days = kwargs.get('trend_days', 5)
        data['ROC'] = talib.ROC(data['Close'], timeperiod=trend_days)
        return data


class StochasticOscillatorIndicator(Indicator):
    def compute(self, data, *args, **kwargs):
        trend_days = kwargs.get('trend_days', 5)
        data['StoK'], data['StoD'] = talib.STOCH(
            data['High'], data['Low'], data['Close'], fastk_period=trend_days, slowk_period=3, slowd_period=3)
        return data


class CCIIndicator(Indicator):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 14)
        data['CCI'] = talib.CCI(data['High'], data['Low'],
                                data['Close'], timeperiod=timeperiod)
        return data


class RSIIndicator(Indicator):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 14)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=timeperiod)
        return data


class VMAIndicator(Indicator):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 20)
        data['VMA'] = talib.MA(data['Volume'], timeperiod=timeperiod)
        return data


class PctChangeIndicator(Indicator):
    def compute(self, data, *args, **kwargs):
        data['pctChange'] = data['Close'].pct_change() * 100
        return data


class ThreeMonthTreasuryYield(Indicator):
    def compute(self, data, *args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        three_month_treasury_yield = yf.download(
            "^IRX", start_date, end_date)["Close"]
        data['3M Treasury Yield'] = three_month_treasury_yield
        return data


class FiveYearTreasuryYield(Indicator):
    def compute(self, data, *args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        five_year_treasury_yield = yf.download(
            "^FVX", start_date, end_date)["Close"]
        data['5Y Treasury Yield'] = five_year_treasury_yield
        return data


class TenYearTreasuryYield(Indicator):
    def compute(self, data, *args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        ten_year_treasury_yield = yf.download(
            "^TNX", start_date, end_date)["Close"]
        data['10Y Treasury Yield'] = ten_year_treasury_yield
        return data


class ThirtyYearTreasuryYield(Indicator):
    def compute(self, data, *args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        thirty_year_treasury_yield = yf.download(
            "^TYX", start_date, end_date)["Close"]
        data['30Y Treasury Yield'] = thirty_year_treasury_yield
        return data
# Add other indicators here as needed


class IndicatorFactory:
    """
    Factory class dedicated to creating various technical indicators.
    """
    @staticmethod
    def get_indicator(indicator_type):
        """
        Retrieve the desired indicator based on the specified type.

        Parameters:
        - indicator_type: Type of technical indicator (e.g., 'Trend', 'MACD').

        Returns:
        - Indicator object corresponding to the specified type.

        Raises:
        - ValueError: If the provided indicator type is not supported.
        """
        indicators = {
            "Trend": TrendIndicator,
            "MACD": MACDIndicator,
            "ROC": ROCIndicator,
            "Stochastic Oscillator": StochasticOscillatorIndicator,
            "CCI": CCIIndicator,
            "RSI": RSIIndicator,
            "VMA": VMAIndicator,
            "PctChange": PctChangeIndicator,
            "3M Treasury Yield": ThreeMonthTreasuryYield,
            "5Y Treasury Yield": FiveYearTreasuryYield,
            "10Y Treasury Yield": TenYearTreasuryYield,
            "30Y Treasury Yield": ThirtyYearTreasuryYield,
            # Add other indicators here as needed
        }
        indicator = indicators.get(indicator_type)
        if indicator is None:
            raise ValueError(f"Invalid indicator type: {indicator_type}")
        return indicator()


class DataCleaner(ABC):
    """Abstract base class for data processors."""
    @abstractmethod
    def check(self, data):
        """Method to check the data for issues."""
        pass

    @abstractmethod
    def clean(self, data):
        """Method to clean the data from identified issues."""
        pass


class MissingDataCleaner(DataCleaner):
    """Concrete class for checking and handling missing data."""
    def check(self, data):
        """Check for missing data in the dataframe."""
        return data.isnull().sum()

    def clean(self, data, strategy='auto'):
        """Handle missing data based on the chosen strategy."""
        if strategy == 'auto':
            while data.iloc[0].isnull().any():
                data = data.iloc[1:]
            data.fillna(method='ffill', inplace=True)

        elif strategy == 'drop':
            data.dropna(inplace=True)

        elif strategy == 'fillna':
            data.fillna(method='ffill', inplace=True)

        elif strategy == 'none':
            pass

        else:
            raise ValueError("Invalid strategy provided.")

        return data


class ScalerFactory:
    """
    Factory class dedicated to creating scalers.
    """
    @staticmethod
    def get_scaler(method):
        if method == 'StandardScaler':
            return StandardScaler()
        elif method == 'MinMaxScaler':
            return MinMaxScaler()
        else:
            raise ValueError(f"Invalid scaler method: {method}.")


class DataProcessorFactory:
    """Factory class to create data processors."""
    @staticmethod
    def create_cleaner(clean_type, *args, **kwargs):
        """Create a data processor based on the provided type."""
        if clean_type == "MissingData":
            return MissingDataCleaner(*args, **kwargs)
        else:
            raise ValueError(f"Processor type {clean_type} not recognized.")

    @staticmethod
    def standardize_data(data, method='StandardScaler'):
        """
        Standardize the data based on the chosen method.

        Parameters:
        - data: The data to be standardized.
        - method: The standardization method.

        Returns:
        - The standardized data.
        """
        scaler = ScalerFactory.get_scaler(method)
        return scaler.fit_transform(data)

    @staticmethod
    def standardize_and_split_data(data, split_ratio=0.7, target_col="Trend", feature_cols=None):
        """Standardize the data and split it into training and testing sets."""
        if not feature_cols:
            feature_cols = data.columns.to_list()
        x_data = data[feature_cols]

        # Generate the one-hot encoding
        y_data = pd.get_dummies(data[target_col], prefix='Trend')

        # Check if the split index is valid
        split_idx = int(len(x_data) * split_ratio)
        if split_idx < 1 or split_idx >= len(x_data):
            raise ValueError(
                "Invalid split ratio leading to incorrect data partitioning.")

        X_test = x_data.iloc[split_idx:]
        y_test = y_data.iloc[split_idx:]
        X_train = x_data.iloc[:split_idx]
        y_train = y_data.iloc[:split_idx]

        return X_train, y_train, X_test, y_test

    @staticmethod
    def prepare_multistep_data(x_data, y_data, look_back, predict_steps, slide_steps=1):
        """
        Prepare the data for multi-step prediction and apply standardization within each sliding window.
        """
        x_date = []
        y_date = []
        x_data_multistep = []
        y_data_multistep = []

        for i in range(0, len(x_data) - look_back - predict_steps + 1, slide_steps):
            x_date.append(x_data.index[i:i + look_back])

            y_date.append(
                x_data.index[i + look_back:i + look_back + predict_steps])

            x_window = x_data.iloc[i:i + look_back].values
            y_window = y_data.iloc[i + look_back:i +
                                   look_back + predict_steps].values

            x_window_standardized = DataProcessorFactory.standardize_data(
                x_window)

            x_data_multistep.append(x_window_standardized)
            y_data_multistep.append(y_window)

        return np.array(x_data_multistep), np.array(y_data_multistep), np.array(x_date), np.array(y_date)


class ModelDataAPI:
    """
    API for fetching, processing, and preparing model data.
    """
    def __init__(self, data=None, start_date=None, end_date=None):
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.trend_method = "MA"
        self.indicators = []
        self.processors = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def set_seed(self, seed_value=42):
        """Set seed for reproducibility."""
        np.random.seed(seed_value)

    def fetch_stock_data(self, stock_symbol, start_date=None, end_date=None):
        """Fetch stock data from Yahoo Finance."""
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        return yf.download(stock_symbol, start=self.start_date, end=self.end_date)

    def add_indicator(self, indicator_type, *args, **kwargs):
        indicator = IndicatorFactory.get_indicator(indicator_type)
        self.data = indicator.compute(self.data, *args, **kwargs)

    def add_data_cleaner(self, clean_type='MissingData', strategy='drop'):
        """Method to check and clean the data using a specific processor."""
        processor = DataProcessorFactory.create_cleaner(clean_type)
        issues = processor.check(self.data)
        self.data = processor.clean(self.data, strategy=strategy)
        return issues

    def process_data(self, split_ratio=0.7, target_col="Trend", feature_cols=None, look_back=64, predict_steps=16, train_slide_steps=1, test_slide_steps=16):
        """
        Use DataProcessorFactory to standardize and split the data, and prepare it for multi-step prediction if required.
        """
        self.X_train, self.y_train, self.X_test, self.y_test = DataProcessorFactory.standardize_and_split_data(
            self.data, split_ratio, target_col, feature_cols)

        if look_back and predict_steps:
            self.X_train, self.y_train, self.train_dates, _ = DataProcessorFactory.prepare_multistep_data(
                self.X_train, self.y_train, look_back, predict_steps, train_slide_steps)
            self.X_test, self.y_test, _, self.test_dates = DataProcessorFactory.prepare_multistep_data(
                self.X_test, self.y_test, look_back, predict_steps, test_slide_steps)

if __name__ == "__main__":
    model_data = ModelDataAPI()
    model_data.set_seed(42)
    start_date = "2001-01-01"
    stop_date = "2021-01-01"
    stock_symbol = "^GSPC"
    model_data.data = model_data.fetch_stock_data(
        stock_symbol, start_date, stop_date)

    indicators = [
        {"type": "Trend", "method": "MA", "oder_days": 20,
            "ma_days": 20, "trend_days": 5},
        {"type": "MACD", "fastperiod": 5, "slowperiod": 10, "signalperiod": 9},
        {"type": "ROC", "trend_days": 5},
        {"type": "Stochastic Oscillator", "trend_days": 5},
        {"type": "CCI", "timeperiod": 14},
        {"type": "RSI", "timeperiod": 14},
        {"type": "VMA", "timeperiod": 20},
        {"type": "PctChange"},
        {"type": "3M Treasury Yield", "start_date": "2001-01-01", "end_date": "2021-01-01"},
        {"type": "5Y Treasury Yield", "start_date": "2001-01-01", "end_date": "2021-01-01"},
        {"type": "10Y Treasury Yield", "start_date": "2001-01-01", "end_date": "2021-01-01"},
        {"type": "30Y Treasury Yield", "start_date": "2001-01-01", "end_date": "2021-01-01"},
    ]  # Add other indicators here as needed

    for indicator_params in indicators:
        indicator_type = indicator_params["type"]
        model_data.add_indicator(indicator_type, **indicator_params)

    issues_detected = model_data.add_data_cleaner("MissingData", strategy='auto')

    split_ratio = 0.7
    target_col = "Trend"
    feature_cols = None  # None means use all columns
    # feature_cols = ['Close']
    look_back = 64  # number of previous days' data to consider
    predict_steps = 16  # number of days to predict in the future
    slide_steps = 1  # sliding window step size

    model_data.process_data(split_ratio=0.7, target_col="Trend", feature_cols=feature_cols, look_back=look_back,
                            predict_steps=predict_steps, train_slide_steps=1, test_slide_steps=predict_steps)
    print(model_data.X_train.shape, model_data.y_train.shape, model_data.X_test.shape, model_data.y_test.shape)
