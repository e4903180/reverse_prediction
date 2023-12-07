import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


class Evaluator:
    def __init__(self):
        pass

    def analyze_results(self, y_test, y_preds, history, online_training_acc, online_training_losses):
        # Confusion Matrix
        precision, recall, accuracy, f1 = self.confusion_matrix(y_preds, y_test)
        print(f'Average Precision: {precision}')
        print(f'Average Recall: {recall}')
        print(f'Average Accuracy: {accuracy}')
        print(f'Average F1 Score: {f1}')

        self.plot_confusion_matrix(y_test, y_preds)
        # Training Loss Curve
        if history:
            self.plot_loss_curve(history)
        # Online Training Loss Curve
        self.plot_online_training_curve(online_training_acc, online_training_losses)

    def confusion_matrix(self, y_preds, y_test):
        # Flatten the 3D tensors for evaluation
        y_test_flat = np.argmax(y_test.reshape(-1, y_test.shape[-1]), axis=1)
        y_preds_flat = np.argmax(y_preds.reshape(-1, y_preds.shape[-1]), axis=1)

        # Calculate evaluation metrics
        precision = precision_score(y_test_flat, y_preds_flat, average='macro')
        recall = recall_score(y_test_flat, y_preds_flat, average='macro')
        accuracy = accuracy_score(y_test_flat, y_preds_flat)
        f1 = f1_score(y_test_flat, y_preds_flat, average='macro')

        return precision, recall, accuracy, f1
    
    def plot_confusion_matrix(self, y_test, y_preds):
        # Convert to class labels if necessary
        y_test = np.argmax(y_test.reshape(-1, y_test.shape[-1]), axis=1)
        y_preds = np.argmax(y_preds.reshape(-1, y_preds.shape[-1]), axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_preds)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')

        # Calculate metrics
        recall = recall_score(y_test, y_preds, average='macro')
        accuracy = accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds, average='macro')

        # Annotate metrics on the plot
        plt.xlabel(f'Predicted\n\nAccuracy: {accuracy:.2f}\nF1 Score: {f1:.2f}')
        plt.ylabel(f'Actual\n\nRecall: {recall:.2f}')
        plt.show()

    def plot_loss_curve(self, history):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot training loss on the first subplot
        ax1.plot(history.history['loss'], color='tab:blue')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(history.history['binary_accuracy'], color='tab:green')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        ax2.grid(True)

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()

    def plot_online_training_curve(self, acc, losses):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy on the first subplot
        ax1.plot(acc, color='tab:red')
        ax1.set_title('Online Training Accuracy')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True)

        # Plot loss on the second subplot
        ax2.plot(losses, color='tab:blue')
        ax2.set_title('Online Training Loss')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        # Adjust the layout
        plt.tight_layout()
        plt.show()

    def perform_backtesting(self, stock_data, pred_trade_signals):
        # Initialize cerebro engine
        cerebro = bt.Cerebro()

        # Create a data feed from stock data
        data_feed = bt.feeds.PandasData(dataname=stock_data)

        # Add data feed to cerebro
        cerebro.adddata(data_feed)

        # Define and add strategy
        class SignalStrategy(bt.Strategy):
            def __init__(self):
                # Map dates to signals for quick lookup
                self.signal_dict = dict((pd.Timestamp(date).to_pydatetime().date(), signal)
                                        for date, signal in zip(pred_trade_signals['Date'], pred_trade_signals['Signal']))

            def next(self):
                # Get the current date
                current_date = self.datas[0].datetime.date(0)

                # Check if there's a signal for this date
                signal = self.signal_dict.get(current_date)

                # Implement your trading logic based on the signal
                if signal == 'Buy':
                    # Buy logic
                    self.buy()
                elif signal == 'Sell':
                    # Sell logic
                    self.sell()

        # Add strategy to cerebro
        cerebro.addstrategy(SignalStrategy)

        # Set initial cash, commission, etc.
        cerebro.broker.setcash(10000.0)
        cerebro.broker.setcommission(commission=0.001)

        # You can add more code here to analyze the results
        # Add analyzers to cerebro
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

        # Run the backtest
        strategies = cerebro.run()

        # Extracting and displaying results
        for strategy in strategies:
            print("Final Portfolio Value: ", strategy.broker.getvalue())
            print("Sharpe Ratio: ", strategy.analyzers.sharpe_ratio.get_analysis())
            print("Drawdown Info: ", strategy.analyzers.drawdown.get_analysis())
            print("Trade Analysis: ", strategy.analyzers.trade_analyzer.get_analysis())
        # Plotting the results
        # cerebro.plot(style='candlestick')

        return strategies
