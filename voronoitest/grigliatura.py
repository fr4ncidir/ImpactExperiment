#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  grigliatura.py
#  
#  Copyright 2018 Francesco Antoniazzi <francesco.antoniazzi@unibo.it>
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

from numpy import array, linspace, zeros, transpose, tile, repeat, argmin, int8, linalg, argmax, sum, diagonal
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from keras.callbacks import TerminateOnNaN
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

def parse_coord_csv(path_to_file):
    from csv import reader
    with open(path_to_file, "r") as matlabfile :
        coord_file = reader(matlabfile)
        for row in coord_file:
            item = []
            for element in row[0].split(" "):
                item.append(float(element))
            yield item
            
def elaborate_infinite_region(vor,region,new_point,step_x,step_y,fine_grain):
    # agricolo, ma funzionante
    if -1 in region:
        polygon = []
        for item in region:
            if item != -1:
                polygon.append(list(vor.vertices[item]))
        if len(polygon) == 2:
            if polygon[0][0] == polygon[1][0]:
                if polygon[0][0]-new_point[0]>0:
                    tba=[[polygon[0][0]-step_x,polygon[0][1]],[polygon[1][0]-step_x,polygon[1][1]]]
                    tba.sort(key=lambda x: x[1],reverse=True)
                else:
                    tba=[[polygon[0][0]+step_x,polygon[0][1]],[polygon[1][0]+step_x,polygon[1][1]]]
                    tba.sort(key=lambda x: x[1],reverse=True)
            else:
                if polygon[0][1]-new_point[1]>0:
                    tba=[[polygon[0][0],polygon[0][1]-step_y],[polygon[1][0],polygon[1][1]-step_y]]
                    tba.sort(key=lambda x: x[0],reverse=True)
                else:
                    tba=[[polygon[0][0],polygon[0][1]+step_y],[polygon[1][0],polygon[1][1]+step_y]]
                    tba.sort(key=lambda x: x[0],reverse=True)
        else:
            if new_point[0]>polygon[0][0]:
                # destra
                if new_point[1]>polygon[0][1]:
                    #alto
                    tba=[   [polygon[0][0],polygon[0][1]+step_y],
                            [polygon[0][0]+step_x,polygon[0][1]+step_y],
                            [polygon[0][0]+step_x,polygon[0][1]]]
                else:
                    #basso
                    tba=[   [polygon[0][0],polygon[0][1]-step_y],
                            [polygon[0][0]+step_x,polygon[0][1]-step_y],
                            [polygon[0][0]+step_x,polygon[0][1]]]
            else:
                # sinistra
                if new_point[1]>polygon[0][1]:
                    #alto
                    tba=[   [polygon[0][0],polygon[0][1]+step_y],
                            [polygon[0][0]-step_x,polygon[0][1]+step_y],
                            [polygon[0][0]-step_x,polygon[0][1]]]
                else:
                    #basso
                    tba=[   [polygon[0][0],polygon[0][1]-step_y],
                            [polygon[0][0]-step_x,polygon[0][1]-step_y],
                            [polygon[0][0]-step_x,polygon[0][1]]]
        polygon.extend(tba)
    else:
        polygon = vor.vertices[region]
        print(polygon)
    return polygon

def get_region_number(centroids,point):
    return argmin(linalg.norm(centroids-point,axis=1))
    
def train_network(angles, ideal_matrix):
    angles_training, angles_test, vec_training, vec_test = train_test_split(
                 angles,                # matlab generated angles
                 ideal_matrix,          # corresponding class label
                 test_size=15,          # % of the dataset for prediction
                 random_state=42)       # seed for random number generator
    
    # create model
    model = Sequential()
    model.add(Dense(500,                    # neuron number of first level
                    input_dim=4,                # 4 angles -> 4 inputs
                    activation="sigmoid")      # activation function: sigmoid, tanh, relu
                )
    model.add(Dense(ideal_matrix.shape[1],     # needing 2 outputs, the second level only has 2 neurons
        activation='softmax'))                  # activation function of the second level
    #compile model
    model.compile(loss='mean_squared_error',            # loss function for training evolution
                  optimizer="adam",               # optimization fuction: sgd, adagrad, adam
                  metrics=['accuracy'])
                  
    history = model.fit(array(angles_training),         # dataset part for training (features: angles)
                        array(vec_training),            # dataset part for training (labels: impact position coord)
                        batch_size=10,                  # dataset subsets cardinality
                        epochs=20,                   # subset repetition in training
                        verbose=1,
                        validation_split=0.1,           # % subset of the training set for validation (leave-one-out,k-fold)
                        callbacks=[])
                        
    return model.predict(array(angles_test)), vec_test
                        

def main(args):
    side = 1
    fine_grain = 25
    
    mygen_x,step_x = linspace(0,side,num=fine_grain,retstep=True)
    mygen_y,step_y = linspace(0,side,num=fine_grain,retstep=True)
    mygen = transpose([tile(mygen_x, len(mygen_y)), repeat(mygen_y, len(mygen_x))])
    print("stepx={}, stepy={}".format(step_x,step_y))

    vor = Voronoi(mygen)
    # voronoi_plot_2d(vor)
    # print("Voronoi verts: ")
    # print(vor.vertices)
    # print("Voronoi regions: ")
    #print(vor.regions)
    
    # for index,item in enumerate(mygen):
        # plt.text(item[0],item[1],index)
    region_number = len(vor.regions)-1
    print("{} regions available".format(region_number))
    
    sensors = array([(1/3,1/3), (1/3,2/3), (2/3,1/3), (2/3,2/3)])*side
    # for s in sensors:
        # plt.plot(s[0], s[1], 'r+')
    
    real_impacts = array(list(parse_coord_csv("../ImpactExperiment/livello_0/iX_ideal.csv")))
    impact_number = len(real_impacts)
    ideal_matrix = zeros((impact_number,region_number),dtype=int8)
    for index,impact in enumerate(real_impacts): 
        # plt.plot(impact[0], impact[1], 'ro')
        point_region = get_region_number(mygen,impact)
        print("Region is: {}".format(point_region))
        ideal_matrix[index][point_region] += 1
        # ridges = numpy.where(vor.ridge_points == point_region)[0]
        # vertex_set = set(array(vor.ridge_vertices)[ridges, :].ravel())
        # region = [x for x in vor.regions if set(x) == vertex_set][0]
        # polygon = elaborate_infinite_region(vor,region,impact,step_x,step_y,fine_grain)
        # plt.fill(*zip(*polygon), color='yellow')
    plt.matshow(ideal_matrix)
    plt.show()
    
    angles = array(list(parse_coord_csv("../ImpactExperiment/livello_0/a_ideal.csv")))
    
    predictions, real_results = train_network(angles,ideal_matrix)
    
    prediction_region = argmax(predictions,axis=1)
    real_regions = argmax(real_results,axis=1)
    
    with open("./confronto.txt","w") as cfr:
        for index in range(len(prediction_region)):
            cfr.write("{} --> {}\n".format(prediction_region[index],real_regions[index]))
    
    confusion_matrix = zeros((region_number,region_number),dtype=int8)
    n_h = 0
    n_v = 0
    n_d = 0
    for index,real_region in enumerate(real_regions):
        confusion_matrix[real_region][prediction_region[index]]+=1
        neighbors = {"h":[], "v":[], "d":[]}
        if real_region % fine_grain != 0:
            neighbors["h"].append(real_region-1)
            if real_region > fine_grain:
                neighbors["d"].append(real_region-1-fine_grain)
            if real_region < fine_grain*(fine_grain-1):
                neighbors["d"].append(real_region-1+fine_grain)
        if ((real_region+1) % fine_grain) != 0:
            neighbors["h"].append(real_region+1)
            if real_region > fine_grain:
                neighbors["d"].append(real_region+1-fine_grain)
            if real_region < fine_grain*(fine_grain-1):
                neighbors["d"].append(real_region+1+fine_grain)
        if real_region > fine_grain:
            neighbors["v"].append(real_region-fine_grain)
        if real_region < fine_grain*(fine_grain-1):
            neighbors["v"].append(real_region+fine_grain)
        
        if prediction_region[index] in neighbors["h"]:
            n_h += 1
        elif prediction_region[index] in neighbors["v"]:
            n_v += 1
        elif prediction_region[index] in neighbors["d"]:
            n_d += 1
    
    print("Confusion matrix statistics: ")
    c_matrix_sum = sum(confusion_matrix)
    dp = sum(diagonal(confusion_matrix))/c_matrix_sum
    print("Diagonal percentage: {}%".format(dp))

    tp = (n_h+n_v+n_d)/c_matrix_sum
    print("H: {}%\tV: {}%\tDiag: {}\tTot: {}".format(n_h/c_matrix_sum,n_v/c_matrix_sum,n_d/c_matrix_sum,tp))
    
    print("Completely wrong predictions: {}".format(1-dp-tp))
    
    plt.matshow(confusion_matrix)
    plt.show()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
