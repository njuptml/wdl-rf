# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:18:13 2017

@author: zqm
"""
import autograd.numpy as np

from wdl import load_data1
from wdl import build_wdl_fingerprint_fun

import pickle

task_params = {'input_name' : 'smiles',
               'data_file'   : 'data.csv'}
               
GPCR=['P08908','P24530','P30542','P30968',
          'P35372','P50406','Q99705','P47871','P08912','P35348','P46663','P51677',
          'P21917','Q9Y5N1','Q99500','Q9Y5Y4','P34995','P35346','P21452']
          
model_params = dict(fp_length=50,    
                    fp_depth=4,      
                    hidden_width=20)

def run_fp(datasmile):  
    print datasmile
    def build_single_weight_fp_experiment(train_data,init_weights,x=0):
        
        fp_depth=x
        hidden_layer_sizes = [model_params['hidden_width']] * fp_depth 
        hidden_arch_params = {'num_hidden_features' : hidden_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1}
        fp_func, conv_parser = build_wdl_fingerprint_fun(**hidden_arch_params)
        datafp=fp_func(init_weights,train_data)
        return datafp[0]
    
    def show_weight_fp_experiment(gpcr):
        train_data= load_data1(gpcr+'.csv', task_params['input_name'])
        train_data=np.hstack((np.array(datasmile),train_data))
        print gpcr
        pkl_file_name=gpcr+'.pkl'
        pick_file=open(pkl_file_name,"rb")
        init_weights=pickle.load(pick_file)
        pick_file.close
        datafp0= build_single_weight_fp_experiment(train_data,init_weights,0)  
        for i in range(1,model_params['fp_depth']):
            print 'FP'+str(i)+':'
            datafpx= build_single_weight_fp_experiment(train_data,init_weights,i)
            print (datafpx-datafp0)
            print 
            datafp0=datafpx 
        datafp1= build_single_weight_fp_experiment(train_data,init_weights,model_params['fp_depth'])         
        print 'FP4:'
        print (datafp1-datafp0)
        print
        print 'FP5:'
        print datafp1
        print        
    for each_gpcr in GPCR:
        show_weight_fp_experiment(each_gpcr)

def main():
    print "Loading data..."
    data= load_data1(task_params['data_file'], task_params['input_name']) 
    for each_smile in data:
        run_fp([each_smile])

    
if __name__ == '__main__':
    main()