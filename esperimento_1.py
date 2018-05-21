#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  esperimento_1.py
#  
#  Copyright 2018 Eugenio <eugenio@eugenio-VirtualBox>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
#import numpy as np
#import matplotlib.pyplot as plt

from keras.callbacks import History
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras import metrics
from sklearn.model_selection import train_test_split
from numpy import array,linalg
import logging
import config 
from experiment_result import *
from time import time
from custom_callbacks import EarlyStopping

logging.basicConfig(format='%(levelname)s %(asctime)-15s %(message)s',level=logging.INFO)
early_stop = []

def my_handler(stop_value):
    early_stop.append(stop_value)
    

def run_experiment(angles,positions,test_size, batch_size, neurons, activation, optimization, epoch):
    global early_stop
    angles_training, angles_test, pos_training, pos_test = train_test_split(
                     angles,                # matlab generated angles
                     positions,             # corresponding coordinates
                     test_size=test_size,   # % of the dataset for prediction
                     random_state=42)       # seed for random number generator

    # create model
    model = Sequential()
    model.add(Dense(neurons,                    # neuron number of first level
                    input_dim=4,                # 4 angles -> 4 inputs
                    activation=activation)      # activation function: sigmoid, tanh, relu
                )
    model.add(Dense(2,                          # needing 2 outputs, the second level only has 2 neurons
        activation=activation)                  # activation function of the second level
        )

    #compile model
    model.compile(loss='mean_squared_error',            # loss function for training evolution
                  optimizer=optimization)               # optimization fuction: sgd, adagrad, adam

    #fit model

    mylog = {}
    previous_early_stop_len = len(early_stop)
    history = model.fit(array(angles_training),         # dataset part for training (features: angles)
                        array(pos_training),            # dataset part for training (labels: impact position coord)
                        batch_size=batch_size,          # dataset subsets cardinality
                        epochs=epoch,                   # subset repetition in training
                        verbose=0,
                        validation_split=0.1,           # % subset of the training set for validation (leave-one-out,k-fold)
                        callbacks=[EarlyStopping(monitor='val_loss',verbose=True,handler=my_handler)])
    
    
    # evaluate the model
    scores = model.evaluate(array(angles_test),         # dataset part for test (features: angles)
                            array(pos_test),            # dataset control of predictions: labels 
                            batch_size=None)            # evaluation batch (None=default=32)

    if (len(early_stop)>previous_early_stop_len):
        stop = early_stop[-1]
        early_stop = []
        return scores,stop
    else:
        return scores,None
    

def main(args):
    logging.info('parsing angles file')
    angles=config.parse_csv('./a.csv')
    positions=config.parse_csv('./iX.csv')
    line_storage = []
    
    af=args['activation-function']
    opt=args['optimizator']
    for ts in range(20,50,20):
        for bs in range(1,40,20):
            for neu in range(20,100,20):
                for epoch in range(10,100,20):
                    logging.info('running experiment with ts={}\tbs={}\tneu={}\taf={}\topt={}\tepoch={}'.format(ts,bs,neu,af,opt,epoch))
                    start_time=time()
                    loss,stop=run_experiment(angles,positions,ts, bs, neu, af, opt, epoch)
                    elapsed_time=(time()-start_time)*1000
                    line_storage.append(ExperimentResult(ts,bs,neu,af,opt,epoch,loss,elapsed_time,stop))
       
    
    line_storage.sort(key = lambda x: x.loss)
    with open('./results.txt','w') as results:
        for item in line_storage:
            results.write(item.toString())
    return 0

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Esperimento Livello 1")
    parser.add_argument("activation-function", choices=['sigmoid','tanh','relu'],
                        help="funzioni di attivazione per la rete test")
    parser.add_argument("optimizator", choices=['sgd', 'adagrad','adam'],
                        help="funzioni di ottimizzazione")
    args = vars(parser.parse_args())

    config.configurations()
    sys.exit(main(args))

