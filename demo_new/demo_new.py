# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:26:58 2017

@author: zqm
"""

import autograd.numpy as np
import autograd.numpy.random as npr

from wdl import load_data
from wdl import build_wdl_deep_net,build_wdl_fingerprint_fun
from wdl import normalize_array,adam
from wdl import build_batched_grad
from wdl.util import rmse,R2,Rs2
from sklearn.ensemble import RandomForestRegressor

from autograd import grad

task_params = {'target_name' : 'STANDARD_VALUE',
               'data_file'   : 'P30968.csv'}
N_train = 1100
N_val   = 112
N_test  = 112

model_params = dict(fp_length=50,    
                    fp_depth=4,      
                    hidden_width=20,  
                    h1_size=100,     
                    layer_weight=0.5,
                    n_estimators=100,
                    max_features='sqrt',
                    L2_reg=np.exp(-2))
train_params = dict(num_iters=250,
                    batch_size=100,
                    init_scale=np.exp(-4),
                    step_size=0.01)

# Define the architecture of the network that sits on top of the fingerprints.
vanilla_net_params = dict(
    layer_sizes = [model_params['fp_length'], model_params['h1_size']],  
    normalize=True, L2_reg = model_params['L2_reg'], nll_func = rmse)

def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params, seed=0,
             validation_smiles=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print "Total number of weights in the network:", num_weights
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    num_print_examples = 100
    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = []
    def callback(weights, iter):
        if iter % 10 == 0:
            print "max of weights", np.max(np.abs(weights))
            train_preds = undo_norm(pred_fun(weights, train_smiles[:num_print_examples]))
            cur_loss = loss_fun(weights, train_smiles[:num_print_examples], train_targets[:num_print_examples])
            training_curve.append(cur_loss)
            print "Iteration", iter, "loss", cur_loss,\
                  "train RMSE", rmse(train_preds, train_raw_targets[:num_print_examples]),\
                  "train R^2" , R2(train_preds, train_raw_targets[:num_print_examples]),\
                  "train Rs^2", Rs2(train_preds, train_raw_targets[:num_print_examples]),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", iter, ":", rmse(validation_preds, validation_raw_targets)
                print "Validation R^2", iter, ":", R2(validation_preds, validation_raw_targets)
                print "Validation Rs^2", iter, ":", Rs2(validation_preds, validation_raw_targets)


    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve


def main():
    print "Loading data..."
    traindata, valdata, testdata = load_data(
        task_params['data_file'], (N_train, N_val, N_test),
        input_name='smiles', target_name=task_params['target_name'])
    train_inputs, train_targets = traindata
    val_inputs,   val_targets   = valdata
    test_inputs,  test_targets  = testdata

    def build_single_weight_fp_experiment(init_weights,x=0):
        fp_depth=x
        hidden_layer_sizes = [model_params['hidden_width']] * fp_depth  
        hidden_arch_params = {'num_hidden_features' : hidden_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1}
        fp_func, conv_parser = build_wdl_fingerprint_fun(**hidden_arch_params)
        trainfp=fp_func(init_weights,train_inputs)
        testfp=fp_func(init_weights,test_inputs) 
        return trainfp,testfp
    
    def build_weight_fp_experiment(init_weight):
        train_x0,test_x0= build_single_weight_fp_experiment(init_weight,0)  
        train_x=model_params['layer_weight']*train_x0
        test_x=model_params['layer_weight']*test_x0
        for i in range(1,model_params['fp_depth']):
            train_x1,test_x1= build_single_weight_fp_experiment(init_weight,i)
            train_x=train_x+model_params['layer_weight']*(train_x1-train_x0)
            test_x=test_x+model_params['layer_weight']*(test_x1-test_x0)
            train_x0=train_x1
            test_x0=test_x1
        train_xx,test_xx= build_single_weight_fp_experiment(init_weight,model_params['fp_depth'])
        train_x=train_x+train_xx-train_x0
        test_x=test_x+test_xx-test_x0
        return train_x,test_x
        

    def run_weight_fp_experiment():
        hidden_layer_sizes = [model_params['hidden_width']] * model_params['fp_depth']
        hidden_arch_params = {'num_hidden_features' : hidden_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1}
        loss_fun, pred_fun, wfp_parser = \
            build_wdl_deep_net(hidden_arch_params, vanilla_net_params, model_params['L2_reg'])
        num_weights = len(wfp_parser)
        predict_func, trained_weights, conv_training_curve = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                     train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets) 
        return trained_weights

    print "Task params", task_params
    print 
    
    print "Starting weight fingerprint experiment..."
    trained_weights = run_weight_fp_experiment()
    train_x,test_x=build_weight_fp_experiment(trained_weights)
    
    train_y,undo_norm1 = normalize_array(train_targets)
    test_y,undo_norm3 = normalize_array(test_targets)
    
    clf = RandomForestRegressor(model_params['n_estimators'],max_features='sqrt')    
    clf = clf.fit(train_x, train_y)
    
    print 
    print "WFP test RMSE:" ,rmse(undo_norm3(clf.predict(test_x)),test_targets),"WFP test R2:",R2(undo_norm3(clf.predict(test_x)),test_targets),\
        "WFP test Rs2:",Rs2(undo_norm3(clf.predict(test_x)),test_targets)
    print 
    

if __name__ == '__main__':
    main()