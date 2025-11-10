from utils.splits import make_or_load_split 
from itertools import product 
from pathlib import Path 

dgp = "blackwell_yamauchi"

rho = [5,10,50] # n/T

params_grid = {
    "n": [200, 500, 1000, 3000], 
    "a": [1,2], 
    "p": [2,4], 
}

train_test_split = 0.8 
split_seed = 42 

T =  {int((1/train_test_split) * n):[int(n/r) for r in rho] for n in params_grid["n"]}
params_grid["n"] = [int(i * (1/train_test_split)) for i in params_grid["n"]] 


for n,p in product(params_grid["n"], params_grid["p"]):
    for t in T[n]:
        _,_ = make_or_load_split(
            dgp=dgp,
            N=n,
            T=t,
            p=p,
            split_seed=split_seed, 
            train= train_test_split,
            val=0.10,
            test=0.10,
            output_dir=Path(f"data/splits/{dgp}/"))