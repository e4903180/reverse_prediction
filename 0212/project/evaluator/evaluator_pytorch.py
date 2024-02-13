import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix
import seaborn as sns
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from io import StringIO
import sys
from pytorchsummary import summary
import os
import json

class Evaluator:
    def __init__(self, params):
        self.params = params
        pass

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
    
    def plot_confusion_matrix(self, y_test, y_preds, save_path='plots/confusion_matrix.png'):
        # Convert to class labels if necessary
        y_test = np.argmax(y_test.reshape(-1, y_test.shape[-1]), axis=1)
        y_preds = np.argmax(y_preds.reshape(-1, y_preds.shape[-1]), axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_preds)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')

        # Calculate metrics
        precision = precision_score(y_test, y_preds, average='macro')
        recall = recall_score(y_test, y_preds, average='macro')
        accuracy = accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds, average='macro')

        # Annotate metrics on the plot
        plt.xlabel(f'Predicted\n\nAccuracy: {accuracy:.2f}\nF1 Score: {f1:.2f}\nPrecision: {precision:.2f}')
        plt.ylabel(f'Actual\n\nRecall: {recall:.2f}')
        plt.savefig(save_path)
        plt.close()
        # plt.show()
        confusion_matrix_text = \
        f'''\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\n
        '''
        return save_path, confusion_matrix_text

    def plot_training_curve(self, history, save_path='plots/training_curve.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss and validation loss
        ax1.plot(history['loss'], label='Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()

        # Plot accuracy and validation accuracy
        ax2.plot(history['binary_accuracy'], label='Accuracy')
        ax2.plot(history['val_binary_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        # plt.show()

        return save_path

    def plot_online_training_curve(self, acc, losses, save_path='plots/online_training_curve.png'):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss on the second subplot
        ax1.plot(losses, color='tab:blue')
        ax1.set_title('Online Training Loss')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy on the first subplot
        ax2.plot(acc, color='tab:red')
        ax2.set_title('Online Training Accuracy')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        # Adjust the layout
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        # plt.show()

        return save_path
    
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
    
    def plot_predictions(self, y_test, y_preds, filter, save_path='plots/predictions.png'):
        # Convert one-hot encoded arrays to integer labels
        y_test_labels = np.argmax(y_test, axis=-1).flatten()
        y_preds_labels = np.argmax(y_preds, axis=-1).flatten()
        if filter != 'False':
            y_preds_labels = self.remove_short_sequences(y_preds_labels.clone(), filter)
        plt.figure(figsize=(32, 6))
        # Plotting y_test
        plt.plot(y_test_labels, label='y_test')

        # Plotting y_preds
        plt.plot(y_preds_labels, label='y_preds')

        # Adding labels and legend
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(fontsize=20)
        plt.savefig(save_path)
        # Display the plot
        plt.close()
        # plt.show()
        return save_path
    
    def plot_trading_signals(self, data, trade_signals, x_start=0, x_stop=-1, save_path='plots/trading_details_kbar.png'):
        stock_data = data[['Open', 'High', 'Low', 'Close']].loc[data.index.isin(trade_signals['Date'])]
        stock_data['pred_signal'] = trade_signals['Signal'].values
    
        fig, ax = plt.subplots(figsize=(32, 6))
        for i in stock_data['pred_signal'].index[x_start:x_stop]:
            self._kbar(stock_data['Open'].loc[i], stock_data['Close'].loc[i], stock_data['High'].loc[i], stock_data['Low'].loc[i], i, ax)

        self._plot_signals(trade_signals, stock_data, x_start, x_stop, ax)
        ax.set_title(f'Trading Details, from {stock_data.index[x_start].date()} to {stock_data.index[x_stop].date()}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_xticks(stock_data.index[x_start:x_stop])
        # ax.set_xticklabels(stock_data.index[x_start:x_stop].strftime('%Y-%m-%d'), rotation=30, ha='right', fontsize=6)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(6)
        plt.grid()
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        # plt.show()

        return save_path

    def _kbar(self, open, close, high, low, pos, ax): # for US stocks
        if close > open: 
            color='green'   # rise               
            height=close - open   
            bottom=open             
        else:                         
            color='red'     # fall            
            height=open - close   
            bottom=close             
        ax.bar(pos, height=height,bottom=bottom, width=0.6, color=color)
        ax.vlines(pos, high, low, color=color)

    def _plot_signals(self, trade_signals, stock_data, x_start, x_stop, ax):
        buy_signals = trade_signals.loc[x_start:x_stop][(trade_signals['Signal'] == 'Buy') | (trade_signals['Signal'] == 'Buy (first)')]
        for i in buy_signals['Date']:
            if i in stock_data.index: 
                ax.scatter(i, stock_data.loc[i, 'Low'] - 50, marker='^', color='green', s=100)

        sell_signals = trade_signals.loc[x_start:x_stop][(trade_signals['Signal'] == 'Sell') | (trade_signals['Signal'] == 'Sell (first)')]
        for i in sell_signals['Date']:
            if i in stock_data.index:
                ax.scatter(i, stock_data.loc[i, 'High'] + 50, marker='v', color='red', s=100)

    def find_closest_date(self, pred_trade_signals, test_trade_signals):
        # Filtering and processing signals
        pred_df_filtered = pred_trade_signals[pred_trade_signals['Signal'].notna() & (pred_trade_signals['Signal'] != '')]
        test_df_filtered = test_trade_signals[test_trade_signals['Signal'].notna() & (test_trade_signals['Signal'] != '')]

        for index in pred_df_filtered.index:
            if '(first)' in pred_df_filtered['Signal'].loc[index] or '(last)' in pred_df_filtered['Signal'].loc[index]:
                pred_df_filtered.loc[index, 'Signal'] = pred_df_filtered['Signal'].loc[index].split()[0]
        for index in test_df_filtered.index:
            if '(first)' in test_df_filtered['Signal'].loc[index] or '(last)' in test_df_filtered['Signal'].loc[index]:
                test_df_filtered.loc[index, 'Signal'] = test_df_filtered['Signal'].loc[index].split()[0]
                
        # Creating a new DataFrame to store the results
        pred_days_difference_results = pred_df_filtered.copy()
        pred_days_difference_results['ClosestDateInTest'] = pd.NaT
        pred_days_difference_results['DaysDifference'] = pd.NA
        
        # Iterating through each row in pred_days_difference_results to find the closest date and days difference
        for index, row in pred_days_difference_results.iterrows():
            signal, pred_date = row['Signal'], row['Date']
            same_signal_df = test_df_filtered[test_df_filtered['Signal'] == signal].copy()

            if not same_signal_df.empty:
                same_signal_df['DateDifference'] = (same_signal_df['Date'] - pred_date)
                closest_date = same_signal_df.loc[same_signal_df['DateDifference'].abs().idxmin()]
                pred_days_difference_results.at[index, 'ClosestDateInTest'] = closest_date['Date']
                pred_days_difference_results.at[index, 'DaysDifference'] = closest_date['DateDifference'].days

        return pred_days_difference_results
    
    def _plot_days_difference_bar_chart(self, pred_days_difference_results, save_path='plots/pred_days_difference_bar_chart.png'):
        # Create bar plot
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(pred_days_difference_results)), pred_days_difference_results['DaysDifference'], color='blue', alpha=0.7)
        plt.title('Bar plot of pred_days_difference_results')
        plt.xlabel('Index')
        plt.ylabel('Difference Value')
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.savefig(save_path)
        plt.close()
        # plt.show()
        return save_path
        
    def perform_backtesting(self, stock_data, trade_signals):
        trade_strategy = self.params['trade_strategy']
        
        buffer = StringIO()
        sys.stdout = buffer

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
                self.signal_dict = \
                    dict((pd.Timestamp(date).to_pydatetime().date(), signal)
                                        for date, signal in zip(trade_signals['Date'], 
                                                                trade_signals['Signal']))

            def log(self, txt, dt=None):
                # Logging function for this strategy
                dt = dt or self.datas[0].datetime.date(0)
                print(f'{dt.isoformat()}, {txt}')

            def next(self):
                # Get the current date
                current_date = self.datas[0].datetime.date(0)
                # Check if there's a signal for this date
                signal = self.signal_dict.get(current_date)
                current_price = self.datas[0].open[0]*1.005

                if trade_strategy == 'single':
                    # Original single share buy/sell logic
                    if signal == 'Buy (first)' or signal == 'Buy (last)':
                        # Buy logic
                        self.buy(size=1)
                        self.log("SINGLE BUY EXECUTED")
                    elif signal == 'Sell (first)' or signal == 'Sell (last)':
                        # Sell logic
                        self.sell(size=1)
                        self.log("SINGLE SELL EXECUTED")
                    elif signal == 'Buy':
                        # Buy logic
                        self.buy(size=2)
                        self.log("DOUBLE BUY EXECUTED")  
                    elif signal == 'Sell':
                        # Sell logic
                        self.sell(size=2)
                        self.log("DOUBLE SELL EXECUTED") 
                        
                elif trade_strategy == 'all':
                    # Buy/Sell as many shares as possible
                    if signal == 'Buy (first)' or signal == 'Buy (last)':
                        cash = self.broker.getcash()
                        size_to_buy = int(cash / current_price)  # Only whole shares
                        # if size_to_buy > 0:
                        self.buy(size=size_to_buy)
                        self.log(f"BUY EXECUTED, size_to_buy:{size_to_buy}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                    elif signal == 'Sell (first)' or signal == 'Sell (last)':
                        size_to_sell = self.getposition(self.datas[0]).size
                        # if size_to_sell > 0:
                        self.sell(size=size_to_sell)
                        self.log(f"SELL EXECUTED, size_to_sell:{size_to_sell}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                    elif signal == 'Buy':
                        # size_to_sell = self.getposition(self.datas[0]).size
                        # # if size_to_sell > 0:
                        # self.sell(size=size_to_sell)
                        # self.log(f"SELL EXECUTED, size_to_sell:{size_to_sell}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                        # size_to_buy = self.getposition(self.datas[0]).size*-1
                        # self.buy(size=size_to_buy)
                        # self.log(f"BUY EXECUTED, size_to_buy:{size_to_buy}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                        cash = self.broker.getcash()
                        size_to_buy = int(cash / current_price)  # Only whole shares
                        # if size_to_buy > 0:
                        self.buy(size=size_to_buy)
                        self.log(f"BUY EXECUTED, size_to_buy:{size_to_buy}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                    elif signal == 'Sell':
                        # cash = self.broker.getcash()
                        # size_to_buy = int(cash / current_price)  # Only whole shares
                        # # if size_to_buy > 0:
                        # self.buy(size=size_to_buy)
                        # self.log(f"BUY EXECUTED, size_to_buy:{size_to_buy}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                        size_to_sell = self.getposition(self.datas[0]).size
                        # if size_to_sell > 0:
                        self.sell(size=size_to_sell)
                        self.log(f"SELL EXECUTED, size_to_sell:{size_to_sell}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")

            def notify_order(self, order):
                if order.status in [order.Completed]:
                    cash = self.broker.getcash()
                    value = self.broker.getvalue()
                    if order.isbuy():
                        self.log(f'BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Commission: {order.executed.comm}, Cash: {cash}, Value: {value}')
                    elif order.issell():
                        self.log(f'SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Commission: {order.executed.comm}, Cash: {cash}, Value: {value}')

        # Add strategy to cerebro
        cerebro.addstrategy(SignalStrategy)
        # Set initial cash, commission, etc.
        cerebro.broker.setcash(10000.0)
        cerebro.broker.setcommission(commission=0.005)
        # You can add more code here to analyze the results
        # Add analyzers to cerebro
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        # Run the backtest
        strategies = cerebro.run()
        backtesting_report = dict()
        # Extracting and displaying results
        strategy = strategies[0]
        backtesting_report['sharpe_ratio'] = strategy.analyzers.sharpe_ratio.get_analysis()
        backtesting_report['drawdown'] = strategy.analyzers.drawdown.get_analysis()
        backtesting_report['trade_analyzer'] = strategy.analyzers.trade_analyzer.get_analysis()
        backtesting_report['final_value'] = cerebro.broker.getvalue()
        backtesting_report['pnl'] = backtesting_report['final_value'] - cerebro.broker.startingcash
        backtesting_report['pnl_pct'] = backtesting_report['pnl'] / cerebro.broker.startingcash * 100.0
        backtesting_report['total_return'] = backtesting_report['final_value'] / cerebro.broker.startingcash

        # Plotting the results
        sys.stdout = sys.__stdout__
        trade_summary = buffer.getvalue()
        buffer.close()
        # img = cerebro.plot(style='candlestick', iplot=False, volume=False, savefig=True)
        # img[0][0].savefig(f'plots/backtesting_kbar.png')
        return strategies, backtesting_report, trade_summary
    
    def generate_model_summary(self, model):
        # Capture the summary output
        # buffer = StringIO()
        # sys.stdout = buffer
        # summary(model=model, input_size=(1, 19, 32))
        # sys.stdout = sys.__stdout__
        # model_summary = buffer.getvalue()
        # buffer.close()
        total = sum([param.nelement() for param in model.parameters()])
        model_summary = f'{model}, \nNumber of parameter: {total}'
        return model_summary

    # TODO:　fix the plot_trading_signals bugs
    def _compile_and_save_report(self, model_summary, training_report, online_training_report, 
                                 confusion_matrix, confusion_matrix_text, 
                                 pred_trade_signals_plot_save_path, test_trade_signals_plot_save_path, predictions_save_path, filtered_predictions_save_path,
                                 pred_days_difference_results, pred_days_difference_bar_chart_save_path, 
                                 backtesting_report, trade_summary, execution_time, report_save_path='report.html', summary_save_path='summary.txt'):
        # display(model_summary)
        # display(confusion_matrix_text)
        # Compile the report. This can be done using a HTML generator.
        backtesting_report_table = pd.DataFrame()
        backtesting_report_table['sharpe_ratio'] = [backtesting_report['sharpe_ratio']['sharperatio']]
        # backtesting_report_table['drawdown'] = [backtesting_report['drawdown']['max']['drawdown']]
        backtesting_report_table['initial_value'] = [10000.0]
        backtesting_report_table['final_value'] = [backtesting_report['final_value']]
        backtesting_report_table['total_return'] = [backtesting_report['total_return']]
        backtesting_report_table['final_value'] = [backtesting_report['final_value']]
        # display(backtesting_report_table)
        backtesting_report = backtesting_report_table.to_html()
        backtesting_report_text = backtesting_report_table.to_csv()
        pred_days_difference_abs_mean = pred_days_difference_results['DaysDifference'].abs().mean()
        pred_days_difference_results = pred_days_difference_results.to_html()
        params_text = json.dumps(self.params, indent=4)
        execution_time_text = f"Execution time: {execution_time} seconds"
        report = f'''
        <html>
            <head>
                <title>Stock Price Prediction Report</title>
            </head>
            <body>
                <h1>Stock Price Prediction Report</h1>
                <h2>Parameters</h2>
                <pre>{params_text}</pre>
                <h2>Model Summary</h2>
                <pre>{model_summary}</pre>
                <h2>Execution time</h2>
                <pre>{execution_time_text}</pre>
                <h2>Training Report</h2>
                <img src="{training_report}" />
                <h2>Online Training Report</h2>
                <img src="{online_training_report}" />
                <h2>Confusion Matrix</h2>
                <pre>{confusion_matrix_text}</pre>
                <img src="{confusion_matrix}" />
                <h2>Predictions Compared With Actual Signals</h2>
                <img src="{predictions_save_path}" />
                <h2>Filtered Predictions Compared With Actual Signals</h2>
                <img src="{filtered_predictions_save_path}" />
                <h2>Predict Trading Signals Difference Bar chart</h2>
                <img src="{pred_days_difference_bar_chart_save_path}" />
                <h2>Predict Trading Signals Difference Details</h2>
                <pre>Average Difference: {pred_days_difference_abs_mean}</pre>
                <pre>{pred_days_difference_results}</pre>
                <h2>Predict Trading Details</h2>
                <img src="{pred_trade_signals_plot_save_path}" />
                <h2>Actual Trading Details</h2>
                <img src="{test_trade_signals_plot_save_path}" />
                <h2>Trade Summary</h2>
                <pre>{backtesting_report}</pre>
                <h2>Trade Details</h2>
                <pre>{trade_summary}</pre>
            </body>
        </html>
        '''

        # Save the report to a file
        with open(report_save_path, 'w') as file:
            file.write(report)
        
        summary = f'''\n{confusion_matrix_text}\n{execution_time_text}\n{backtesting_report_text}
        '''
        
        with open(summary_save_path, 'w') as file:
            file.write(summary)

    def generate_report(self, model, y_test, y_preds, history, 
                            online_training_acc, online_training_losses,
                            stock_data, pred_trade_signals, test_trade_signals, execution_time, save_path_root, x_start=0, x_stop=-1):
            model_summary = self.generate_model_summary(model)
            if history:
                training_report = self.plot_training_curve(history, save_path=os.path.join(save_path_root, self.params['plot_training_curve_save_path']))
            else:
                training_report = ''

            online_training_report = self.plot_online_training_curve(online_training_acc, 
                                            online_training_losses, save_path=os.path.join(save_path_root, self.params['online_training_curve_save_path']))
            confusion_matrix, confusion_matrix_text = self.plot_confusion_matrix(y_test, y_preds, save_path=os.path.join(save_path_root, self.params['confusion_matrix_save_path']))
            predictions_save_path = self.plot_predictions(y_test, y_preds, False, save_path=os.path.join(save_path_root, self.params['predictions_save_path']))
            filtered_predictions_save_path = self.plot_predictions(y_test, y_preds, filter=self.params['filter'], save_path=os.path.join(save_path_root, self.params['filtered_predictions_save_path']))
            pred_trade_signals_plot_save_path = self.plot_trading_signals(stock_data, pred_trade_signals, x_start, x_stop, save_path=os.path.join(save_path_root, self.params['pred_trade_signals_plot_save_path']))
            test_trade_signals_plot_save_path = self.plot_trading_signals(stock_data, test_trade_signals, x_start, x_stop, save_path=os.path.join(save_path_root, self.params['test_trade_signals_plot_save_path']))
            pred_days_difference_results = self.find_closest_date(pred_trade_signals, test_trade_signals)
            pred_days_difference_bar_chart_save_path = self._plot_days_difference_bar_chart(pred_days_difference_results, save_path=os.path.join(save_path_root, self.params['pred_days_difference_bar_chart_save_path']))
            backtest_results, backtesting_report, trade_summary = self.perform_backtesting(stock_data, pred_trade_signals)
            self._compile_and_save_report(model_summary, training_report, online_training_report, confusion_matrix, 
                                          confusion_matrix_text, pred_trade_signals_plot_save_path, test_trade_signals_plot_save_path, predictions_save_path, filtered_predictions_save_path,
                                          pred_days_difference_results, pred_days_difference_bar_chart_save_path, backtesting_report, trade_summary, execution_time,
                                          report_save_path=os.path.join(save_path_root, self.params['report_save_path']), summary_save_path=os.path.join(save_path_root, self.params['summary_save_path']))
