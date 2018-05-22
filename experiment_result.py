#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  experiment_result.py
#  
#  Copyright 2018 Francesco Antoniazzi <francesco.antoniazzi1991@gmail.com>
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


class ExperimentResult:
    """A class containing the result of an experiment"""
    
    def __init__(self,ts,bs,neu,af,opt,epoch,loss,elapsed_time,early_stop):
        self._ts = ts
        self._bs = bs
        self._neu = neu
        self._af = af
        self._opt = opt
        self._epoch = epoch
        self._loss = loss
        self._elapsed_time = elapsed_time
        self._early_stop = early_stop
        
    @property
    def ts(self):
        return self._ts
        
    @property
    def bs(self):
        return self._bs
        
    @property
    def neu(self):
        return self._neu
        
    @property
    def af(self):
        return self._af
        
    @property
    def opt(self):
        return self._opt
        
    @property
    def epoch(self):
        return self._epoch
        
    @property
    def loss(self):
        return self._loss
        
    @property
    def elapsed_time(self):
        return self._elapsed_time
        
    @property
    def early_stop(self):
        return self._early_stop
        
    def toString(self):
        return "ts={}\tbs={}\tneu={}\taf={}\topt={}\tepoch={}\tloss={}\telapsed={}\tearly_stop={}\n".format(
            self.ts,
            self.bs,
            self.neu,
            self.af,
            self.opt,
            self.epoch,
            self.loss,
            self.elapsed_time,
            self.early_stop)
            
    def toList(self):
        return [self.ts,
            self.bs,
            self.neu,
            self.af,
            self.opt,
            self.epoch,
            self.loss,
            self.elapsed_time,
            self.early_stop]
