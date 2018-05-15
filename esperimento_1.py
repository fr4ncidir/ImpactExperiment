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

from keras.layers import Dense,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from numpy import array
import logging
import config 
from experiment_result import *
from time import time

logging.basicConfig(format='%(levelname)s %(asctime)-15s %(message)s',level=logging.INFO)

def run_experiment(angles,positions,test_size, batch_size, neurons, activation, optimization, epoch):

    angles_training, angles_test, pos_training, pos_test =train_test_split(angles, positions, test_size= test_size, random_state=42)

    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=4, activation=activation))
    model.add(Dense(2, activation=activation))

    #compile model
    model.compile(loss='mean_squared_error', optimizer=optimization, metrics=['mse','mse'])

    #fit model

    history=model.fit(array(angles_training), array(pos_training), batch_size= batch_size, epochs=epoch, verbose=0, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss')])

    #config.plot_history(history)

    # evaluate the model
    scores = model.evaluate(array(angles_test), array(pos_test),batch_size=batch_size)
    print(scores)
    loss=scores[1]
    accu=scores[2]*100
    #print("\nTest loss: " , scores[0])
    #print("\nTest accuracy: %.2f%%" %  (scores[1]*100))

    return (loss,accu)
    

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
                    (loss,accu)=run_experiment(angles,positions,ts, bs, neu, af, opt, epoch)
                    elapsed_time=(time()-start_time)*1000
                    line_storage.append(ExperimentResult(ts,bs,neu,af,opt,epoch,loss,accu,elapsed_time))
       
    
    line_storage.sort(key = lambda x: x.accu, reverse=True)
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

