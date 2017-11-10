# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:02:37 2017

@author: zqm
"""

import autograd.numpy as np

from wdl import load_data1,load_data2
from wdl import build_wdl_fingerprint_fun
from sklearn.externals import joblib
import pickle

task_params = {'input_name' : 'smiles',
               'data_file'   : 'data.csv'}
               
GPCR=['P08908','P24530','P30542','P30968',
          'P35372','P50406','Q99705','P47871','P08912','P35348','P46663','P51677',
          'P21917','Q9Y5N1','Q99500','Q9Y5Y4','P34995','P35346','P21452']

model_params = dict(fp_length=50,    
                    fp_depth=4,      
                    hidden_width=20,  
                    h1_size=100,     
                    layer_weight=0.5)


def main():
    print "Loading data..."
    data= load_data1(task_params['data_file'], task_params['input_name'])
    parameter=load_data2('parameter.csv','mean','std')
    datafp=[]

    def build_single_weight_fp_experiment(length,train_data,init_weights,x=0):
        hidden_layer_sizes = [model_params['hidden_width']] * x  
        hidden_arch_params = {'num_hidden_features' : hidden_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1}
        fp_func, conv_parser = build_wdl_fingerprint_fun(**hidden_arch_params)        
        trainfp=fp_func(init_weights,train_data)
        return trainfp[:length]
    
    def build_weight_fp_experiment(gpcr):
        train_data= load_data1(gpcr+'.csv', task_params['input_name'])
        train_data=np.hstack((data,train_data))
        pkl_file_name=gpcr+'.pkl'
        pick_file=open(pkl_file_name,"rb")
        init_weights=pickle.load(pick_file)
        pick_file.close
        train_x0= build_single_weight_fp_experiment(len(data),train_data,init_weights,0)  
        train_x=model_params['layer_weight']*train_x0
        for i in range(1,model_params['fp_depth']):
            train_x1= build_single_weight_fp_experiment(len(data),train_data,init_weights,i)
            train_x=train_x+model_params['layer_weight']*(train_x1-train_x0)
            train_x0=train_x1
        train_xx= build_single_weight_fp_experiment(len(data),train_data,init_weights,model_params['fp_depth'])
        train_x=train_x+train_xx-train_x0
        datafp.append(train_x)
    
    def restore_function(X,mean,std):
        rfdata=-(X * std + mean)        
        return np.power(np.array([10]),rfdata)
    
    for i in range(len(GPCR)):
        print 'activity on '+GPCR[i]
        build_weight_fp_experiment(GPCR[i])
        pklfile=GPCR[i]+'_model.pkl'
        clf=joblib.load(pklfile)
        print restore_function(clf.predict(datafp[i]),parameter[0][i],parameter[1][i])
        print         

if __name__ == '__main__':
    main()