#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  config.py
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
import csv

def parse_csv (path_to_file) :
    with open(path_to_file, 'r') as matlabfile :
        reader=csv.reader(matlabfile)
        output=[]
        for row in reader:
            item=[]
            for element in row[0].split(' '):
                item.append(float(element))
            output.append(item)
    return output
	
def configurations ():
    import numpy as np
    import os
    import warnings
    import tensorflow as tf
    import random as rn
    from keras import backend as bK

    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    bK.set_session(sess)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
    warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def plot_history(history):
    import matplotlib.pyplot as plt
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def printResults(fileName,results,sTime,eTime,af,opt,approx):
    import prettytable
    from datetime import datetime as dt
    prettyResults = prettytable.PrettyTable(["index","testsize","batchsize","neurons","epoch","loss","time [ms]","early stop"])
    for index,item in enumerate(results):
        prettyResults.add_row(item.toList(index))
    with open(fileName,'w') as _results:
        _results.write("Start Time: {}\nEnd Time: {}\nElapsed: {} [s]\nActivation:\t{}\nOptimizer:\t{}\nApproximation:\t{}\n\n".format(
            dt.fromtimestamp(sTime).strftime("%H:%M:%S %d-%m-%Y"),
            dt.fromtimestamp(eTime).strftime("%H:%M:%S %d-%m-%Y"),
            eTime-sTime,
            af,
            opt,
            approx))
        _results.write(str(prettyResults))
