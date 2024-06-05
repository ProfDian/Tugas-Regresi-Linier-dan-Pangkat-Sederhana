import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

banner = """
====================================================
||          Regression Implementation Program     ||
||         Linear and Power Law Methods           ||
||                                                ||
||          Abdul Fattah Rahmadiansyah            ||
||               21120122120028                   ||
====================================================
"""

print(banner)

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.rms_error = None
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def calculate_rms_error(self, y_true, y_pred):
        self.rms_error = np.sqrt(mean_squared_error(y_true, y_pred))
        return self.rms_error

class PowerLawModel:
    def __init__(self):
        self.params = None
        self.rms_error = None
    
    def fit(self, X, y):
        def power_law(x, a, b):
            return a * np.power(x, b)
        self.params, _ = curve_fit(power_law, X.flatten(), y.values, p0=[1, 1])
    
    def predict(self, X):
        a, b = self.params
        return a * np.power(X.flatten(), b)
    
    def calculate_rms_error(self, y_true, y_pred):
        self.rms_error = np.sqrt(mean_squared_error(y_true, y_pred))
        return self.rms_error

class RegressionComparison:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X = self.data[['Hours Studied']].values
        self.y = self.data['Performance Index']
        self.linear_model = LinearRegressionModel()
        self.power_model = PowerLawModel()
    
    def fit_models(self):
        self.linear_model.fit(self.X, self.y)
        self.power_model.fit(self.X, self.y)
    
    def predict_and_evaluate(self):
        # Linear Model
        y_pred_linear = self.linear_model.predict(self.X)
        rms_linear = self.linear_model.calculate_rms_error(self.y, y_pred_linear)
        
        # Power Law Model
        y_pred_power = self.power_model.predict(self.X)
        rms_power = self.power_model.calculate_rms_error(self.y, y_pred_power)
        
        return rms_linear, rms_power
    
    def plot_results(self, rms_linear, rms_power):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot linear regression
        x_range_linear = np.linspace(1, 9, 100).reshape(-1, 1)
        y_predict_linear_range = self.linear_model.predict(x_range_linear)
        axs[0].scatter(self.X, self.y, color='blue', label='Data', s=0.8)
        axs[0].plot(x_range_linear, y_predict_linear_range, color='red', label='Linear Regression')
        axs[0].set_xlabel('Hours Studied')
        axs[0].set_ylabel('Performance Index')
        axs[0].set_title(f'RMS Error Linear: {rms_linear:.5f}')
        axs[0].legend()
        axs[0].xaxis.set_ticks(np.arange(1, 10, 1))
        axs[0].yaxis.set_ticks(np.arange(self.y.min(), self.y.max() + 1, 5))

        # Plot power law regression
        x_range_power = np.linspace(1, 9, 100)
        y_predict_power_range = self.power_model.predict(x_range_power)
        axs[1].scatter(self.X, self.y, color='blue', label='Data', s=0.8)
        axs[1].plot(x_range_power, y_predict_power_range, color='orange', label='Power Law Regression')
        axs[1].set_xlabel('Hours Studied')
        axs[1].set_ylabel('Performance Index')
        axs[1].set_title(f'RMS Error Power Law: {rms_power:.5f}')
        axs[1].legend()
        axs[1].xaxis.set_ticks(np.arange(1, 1, 1))
        axs[1].yaxis.set_ticks(np.arange(self.y.min(), self.y.max() + 1, 5))

        # Add main title for the entire figure
        fig.suptitle('Linear and Power Law Regression Results', fontsize=16)

        # Show plot in the middle of the screen
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

def main():
    data_path = 'Student_Performance.csv'
    comparison = RegressionComparison(data_path)
    comparison.fit_models()
    rms_linear, rms_power = comparison.predict_and_evaluate()
    comparison.plot_results(rms_linear, rms_power)

    print(f'RMS Error Linear: {rms_linear:.5f}')
    print(f'RMS Error Power Law: {rms_power:.5f}')

    if rms_linear < rms_power:
        print('Linear Regression provides less lower RMS error.')
    else:
        print('Power Law Regression provides more RMS error.')

if __name__ == "__main__":
    main()
