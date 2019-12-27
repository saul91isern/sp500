from sklearn import preprocessing
from statsmodels import regression
import statsmodels.api as sm
import numpy as np

class DataSet:
    def calculate_alpha_and_beta(self, x, y, periods):
        x_values = x['Adj Close'].pct_change()[1:].values
        y_values = y['Adj Close'].pct_change()[1:].values
        
        betas = [None for index in range(periods + 1)]
        alphas = [None for index in range(periods + 1)]
        
        for index in range(len(x_values) - periods):
            X = x_values[index: (index + periods)]
            Y = y_values[index: (index + periods)]

            X = sm.add_constant(X)
            model = regression.linear_model.OLS(Y, X).fit()
            
            alphas.append(model.params[0])
            betas.append(model.params[1])

        y['alpha'] = np.array(alphas) 
        y['beta'] = np.array(betas) 
        
        return y

    def get_technical_indicators(self, dataset, spy):
        # Assuming 0 for now
        daily_rf_rate = 0
        # Create 7 and 21 days Moving Average
        dataset['ma7'] = dataset['Adj Close'].rolling(window=7).mean()
        dataset['ma21'] = dataset['Adj Close'].rolling(window=21).mean()
        dataset['SMA100'] = dataset['Adj Close'].rolling(window=100).mean()

        # Create MACD
        dataset['26ema'] = dataset['Adj Close'].ewm(span=26, adjust=False).mean()
        dataset['12ema'] = dataset['Adj Close'].ewm(span=12, adjust=False).mean()
        dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

        # Create Bollinger Bands
        dataset['20sd'] = dataset['Adj Close'].rolling(20).std()
        dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
        dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)

        # Create Exponential moving average
        dataset['ema'] = dataset['Adj Close'].ewm(span=100).mean()
        dataset['ROC'] = ((dataset['Adj Close'] - dataset['Adj Close'].shift(100)) / dataset['Adj Close'].shift(100)) * 100
        dataset['SPY_SMA_100'] = spy['Adj Close'].rolling(window=100).mean()
        
        # Compute Volatility on spy and index 
        dataset['SPY_Vol'] = np.log(spy['Adj Close'] / spy['Adj Close'].shift(1)).rolling(window=60).std() * np.sqrt(252)
        dataset['Log_Returns'] = np.log(dataset['Adj Close'] / dataset['Adj Close'].shift(1))
        dataset['Vol'] = dataset['Log_Returns'].rolling(window=60).std() * np.sqrt(252)
        dataset['Sharpe_Ratio'] = (dataset['Log_Returns'].rolling(window=60).mean() - daily_rf_rate) / dataset['Vol']
        
        return dataset
    
class PreProcessing:
    def split_data(self, data):
        total_length = len(data)
        train_index = round(0.8 * total_length)
        
        train = data[:int(train_index), :]
        test = data[int(train_index):, :]
        
        return train, test

    def normalize_data(self, train, test):
        normalizer = preprocessing.Normalizer()
        train = normalizer.fit_transform(train)
        test = normalizer.transform(test)        
        return train, test
    
    def formate_data(self, data, x_len, y_len):
        x_s = []
        y_s = []
        
        for index in range(len(data) - (x_len + y_len)):
            # We delete the close price from the xs
            x_s.append(data[index: (index + x_len), :-1])
            # We just want the close price as target variable
            y_s.append(data[(index + x_len):(index + x_len + y_len), -1])
        
        x_s = np.array(x_s)
        y_s = np.array(y_s)
        
        return x_s, y_s