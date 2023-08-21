import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:

    def __init__(self):
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        
        if Z_0 == None:
            Z_0 = self.Z0[w-1]
        if L == None:
            L = self.L
            
        predicted_score = Z_0 * (1 - np.exp(-L * X / Z_0))
        return predicted_score
        

    def calculate_loss(self, Params, X, Y, W) -> float:
        self.Z0,self.L = Params[:-1],Params[-1]
        sum_squared_error = 0
        for i in range(len(W)):
            predicted_score = self.get_predictions(X[i],self.Z0[W[i]-1],W[i],self.L)
            squared_error = (predicted_score - Y[i]) ** 2
            sum_squared_error += squared_error
        mean_squared_error = sum_squared_error / len(W)
        return mean_squared_error
        
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(data_path)
    return data



def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    
    data = data[data['Innings'] == 1]
    data = data[['Over','Runs.Remaining','Wickets.in.Hand']]
    data['Overs.Remaining'] = 50 - data['Over']
    data = data.dropna()
    return data



def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    
    
    overs_remaining = data['Overs.Remaining'].values
    runs_remaining = data['Runs.Remaining'].values
    wickets_in_hand = data['Wickets.in.Hand'].values
    
    for wickets in range(1,11):
        data_for_each_wicket= data[data['Wickets.in.Hand'] == wickets]
        model.Z0[wickets-1] = np.mean(data_for_each_wicket['Runs.Remaining'].values)
        
    model.L = 0.035 * model.Z0[9]

    Parameters = model.Z0+[model.L]
    sp.optimize.minimize(model.calculate_loss,Parameters,args=(overs_remaining,runs_remaining,wickets_in_hand),method='L-BFGS-B')
        
    return model


def plot(model: DLModel, plot_path: str) -> None:
    plt.figure(1)
    plt.xlim((0, 50))
    plt.ylim((0, 250))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 50, 100, 150, 200, 250])
    plt.xlabel('Overs remaining')
    plt.ylabel('Average runs obtainable')
    colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff70', '#ffc0cb', '#00f220', '#23f000', '#555b65', '#999e45', '#222a55']
    overs_to_go=np.zeros((51))
    for over_number in range(51):
        overs_to_go[over_number]=over_number
    for wicket in range(10):
        predicted_runs = model.get_predictions(overs_to_go,model.Z0[wicket],wicket+1,model.L)
        plt.plot(overs_to_go, predicted_runs, c=colors[wicket], label='Z(u,' + str(wicket + 1) + ')')
        plt.legend()    
    
    plt.savefig(plot_path)  
    plt.close()


def print_model_params(model: DLModel) -> List[float]:
    print('Z0 Parameters (Z0(1),Z0(2),...,Z0(10)) values are : \n',model.Z0)
    print('L Parameter value is : ',model.L)
    return model.Z0 + [model.L]
    


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    
    overs_remaining = data['Overs.Remaining'].values
    wickets_in_hand = data['Wickets.in.Hand'].values
    runs_remaining = data['Runs.Remaining'].values
    predicted_runs = model.get_predictions(overs_remaining, w=wickets_in_hand)
    squared_errors = (predicted_runs - runs_remaining) ** 2
    mean_squared_error = np.mean(squared_errors)
    print('Mean Squared Loss is : ',mean_squared_error)
    variance_runs_remaining = np.var(runs_remaining)
    normalized_squared_error = mean_squared_error / variance_runs_remaining
    print('Normalized Squared Loss is  : ',normalized_squared_error)
    return normalized_squared_error



def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data

    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    
    
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)