#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  esperimento_1.py
#  
#  Copyright 2018   Francesco Antoniazzi <francesco.antoniazzi@unibo.it>
#                   Eugenio Rossini <eugenio.rossini@studio.unibo.it>
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

from keras.callbacks import TerminateOnNaN
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from numpy import array
import logging
import config 
from experiment_result import *
from time import time
from custom_callbacks import EarlyStopping
from math import isnan

logging.basicConfig(format='%(levelname)s %(asctime)-15s %(message)s',level=logging.INFO)

def run_experiment(angles_training,angles_test,pos_training,pos_test, batch_size, neurons, activation, optimization, epoch):
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
    earlystop_callback = EarlyStopping(monitor='val_loss',    
                                 patience=3,
                                 verbose=True)
    history = model.fit(array(angles_training),         # dataset part for training (features: angles)
                        array(pos_training),            # dataset part for training (labels: impact position coord)
                        batch_size=batch_size,          # dataset subsets cardinality
                        epochs=epoch,                   # subset repetition in training
                        verbose=0,
                        validation_split=0.1,           # % subset of the training set for validation (leave-one-out,k-fold)
                        callbacks=[earlystop_callback,TerminateOnNaN()])
    
    # evaluate the model
    scores = model.evaluate(array(angles_test),         # dataset part for test (features: angles)
                            array(pos_test),            # dataset control of predictions: labels 
                            verbose=0,
                            batch_size=None)            # evaluation batch (None=default=32)
    stop = earlystop_callback.get_stopped_epoch()
    return scores,stop,history


def main(args):
    af=args['activation-function']
    opt=args['optimizator']
    level=["ideal","noref","ref"].index(args['approximation'])
    
    logging.info('parsing angles file')
    angles=config.parse_csv('./livello_{}/a.csv'.format(level))
    positions=config.parse_csv('./livello_{}/iX.csv'.format(level))
    line_storage = []
    resultsFileName = "temp.txt"
    
    simulation_tStart = time()
    try:    
        for ts in [25,30,35]:
            angles_training, angles_test, pos_training, pos_test = train_test_split(
                 angles,                # matlab generated angles
                 positions,             # corresponding coordinates
                 test_size=ts,          # % of the dataset for prediction
                 random_state=42)       # seed for random number generator
            for bs in [1,int(ts/2),ts]:
                for neu in range(4,20,4):
                    for epoch in [5,10,50]:
                        logging.info('running experiment with ts={}\tbs={}\tneu={}\taf={}\topt={}\tepoch={}'.format(ts,bs,neu,af,opt,epoch))
                        sTime=time()
                        loss,stop,history=run_experiment(angles_training, angles_test, pos_training, pos_test, bs, neu, af, opt, epoch)
                        if isnan(loss):
                            logging.warning("Loss = NaN detected")
                        deltaT=(time()-sTime)*1000
                        line_storage.append(ExperimentResult(ts,bs,neu,epoch,loss,deltaT,stop,history))
        resultsFileName = './e1/livello_{}/results_{}_{}_{}.txt'.format(level,af,opt,level)
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt")
        resultsFileName = './e1/livello_{}/results_{}_{}_{}_keyboard.txt'.format(level,af,opt,level)
    finally:
        line_storage.sort(key = lambda x: x.loss)
        config.printResults(resultsFileName,line_storage,simulation_tStart,time(),af,opt,args["approximation"])
    
    if args["p"]:
        print("Plot best experiment loss history: \n{}".format(line_storage[0].toString()))
        config.plot_history(line_storage[0].history)
    
    return 0 

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Esperimento Livello 1")
    parser.add_argument("activation-function", choices=['sigmoid','tanh','relu'],
                        help="funzioni di attivazione per la rete test")
    parser.add_argument("optimizator", choices=['sgd', 'adagrad','adam'],
                        help="funzioni di ottimizzazione")
    parser.add_argument("approximation", choices=["ideal","noref","ref"],
                        help="Approssimazione: ideale, senza riflessioni, con riflessioni",
                        default=0)
    parser.add_argument("-p", action="store_true",
                        help="Alla fine della simulazione avvia la procedura di plot della loss")
    args = vars(parser.parse_args())

    #print(args)
    config.configurations()
    sys.exit(main(args))

