Section 4 evaluates the attack in two settings: fixed privacy budget and fixed number of queries.

To run the fixed budget attack, 
use command 'python attack_fixed_budget.py --dataset=mnist --histogram=histograms/mnist/sample-n.npy' or 
            'python attack_fixed_budget.py --dataset=svhn --histogram=histograms/svhn/sample-n.npy'
replace n with a number from 1,2,...,15.

To run the fixed number of queries attack,
use command 'python attack_fixed_num_queries.py --dataset=mnist --histogram=histograms/mnist/sample-n.npy --sigma=m' or 
            'python attack_fixed_num_queries.py --dataset=svhn --histogram=histograms/svhn/sample-n.npy --sigma=m'
replace n with a number from 1,2,...,15, and
replace m with any noise scale size, in the paper we use m=40,60,80,100.