# NeuralMD

## Findings on Neural ODE/SDE Implementation 

## Advantages of Julia over Python for Neural ODE/SDEs

Neural ODEs and SDEs can suffer from performance issues due to computational cost of the ODE/SDE solver. One way to solve this issue is to use the programming language Julia to implement the neural ODE/SDE over Python. On the flip side, Python is used more in the AI community and Julia has less packages than Python. 

Here are some recent benchmarks stats comparing Python and Julia:

* Julia's DiffEqFlux.jl is **30,000 times** [faster than torchdiffeq for solving ODEs](https://gist.github.com/ChrisRackauckas/cc6ac746e2dfd285c28e0584a2bfd320)
* Julia's DiffEqFlux.jl is **100 times** [faster than torchdiffeq for training NeuralODEs](https://gist.github.com/ChrisRackauckas/4a4d526c15cc4170ce37da837bfc32c4)
* Julia's DiffEqFlux.jl is **1000 times** [faster than torchsde](https://gist.github.com/ChrisRackauckas/6a03e7b151c86b32d74b41af54d495c6) for training NeuralSDEs