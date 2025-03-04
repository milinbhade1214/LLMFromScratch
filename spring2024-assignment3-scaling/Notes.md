# Scaling Laws

Given: fixed compute budget (FLOPs)
Lowest training loss


scaling laws --> 
training loss - (model size, amount of compute used)

FLOP budget = 1e19
hyperparam (# of layers, 
            emb size,
            # of heads,
            batch size,
            learning rate)
            +
            (training FLOPs) + training tokens

API --> training loss

fitting of scaling laws: 2e18 (20% )

## To submit
Predicted optimal model size
training hyperparams to use 
models predicted training loss


## Maths
Compute Budget = f(N, D) = 6 N D
N = No of params in model
D = tokens in dataset




