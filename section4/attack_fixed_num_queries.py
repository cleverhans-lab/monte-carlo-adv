from attack import *
from privacy_analysis import *

import sys
import argparse

def main():
    # parser initialization
    parser = argparse.ArgumentParser(description='load canonical pate parameters, choose mnist or svhn')
    parser.add_argument('--dataset', type=str, help='mnist or svhn', default='mnist')
    parser.add_argument('--sigma', type=int, help='noise scale', default='40')
    parser.add_argument('--histogram', type=str, help='histogram location', default='histograms/mnist/sample-1.npy')

    # get input
    args = parser.parse_args()
    dataset = args.dataset
    svhn250 = np.load(args.histogram)
    sigma_gnmax = args.sigma
    #buget is unlimited when we fix the number of queries
    budget = 100000
    #default mnist
    threshold  = 200
    sigma_threshold=150
    delta=1e-5
    if dataset == 'svhn':
        threshold  = 300
        sigma_threshold=200
        delta=1e-6
        
    c=count(svhn250)
    d = np.tile(c,(10000,1))
    privacy_cost = analyze_multiclass_confident_gnmax(
                    votes=np.array(d), 
                    threshold=threshold, 
                    sigma_threshold=sigma_threshold, 
                    sigma_gnmax=sigma_gnmax, 
                    budget=budget, 
                    delta=delta,
                    file='file'+str(svhn250)+'.txt',
                    show_dp_budget='disable', 
                    args=None)[1][-1]
    print('privacy cost of 10000 queries is '+str(privacy_cost))
    final_error, training_error = gradient_descent(svhn250, sigma_gnmax, 10000, 0)
        

if __name__ == "__main__":
    main()