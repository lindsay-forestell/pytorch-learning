# Unit 5: Deep Networks

**Dropout**

Regularziation technique to help with overfitting

Remove some random subset of neurons each epoch when training

Include all neurons when evaluating

Performs better on validation data

<img src = "https://cdn-images-1.medium.com/max/1600/1*iWQzxhVlvadk6VAJjsgXgg.png" width = 600, align = "center">

    class Net(nn.Module):
    def __init__(self,in_size,n_hidden,out_size,p=0):
        super(Net,self).__init__()
        self.drop=nn.Dropout(p=p) --------------------------> Built in PyTorch Dropout object
        self.linear1=nn.Linear(in_size,n_hidden)
        self.linear2=nn.Linear(n_hidden,n_hidden)
        self.linear3=nn.Linear(n_hidden,out_size)
    def forward(self,x):
        x=torch.relu(self.linear1(x)) ----------------------> I think that you can drop out before or after relu?
        x=self.drop(x)                ----------------------> Eg x = torch.relu(self.drop(self.linear1(x)))
        x=torch.relu(self.linear2(x))
        x=self.drop(x)
        x=self.linear3(x)
        return x
        
     model = Net(in,H,out,p=0.1) ----------------------------> Create a model with probability p of killing a neuron
     model = nn.Sequential(nn.Linear(in,H), -----------------> Can go Linear->Drop->ReLU or Linear->ReLU->Drop
                           nn.Drop(p),      -----------------> Typically do non-linear -> dropout
                           nn.ReLU(),
                           nn.Linear(H,H),
                           nn.Drop(p),
                           nn.ReLU(),
                           nn.Linear(H,out))
                           
    model.train() -------------------------------------------> Default, but good practice to always set this
    ...train...   -------------------------------------------> Implementation note: PyTorch normalizes during training
    model.eval() --------------------------------------------> Will switch off of training mode to use all the neurons
                 --------------------------------------------> Always use this when using validation data

**Initialization**

Choosing good inputs can increase validation accuracy

Always initializes from random uniform distribution, but with varying widths. (width = 2L)

Uniform: L = 1

Default: L = 1/Sqrt(Lin) ----------------> Good for use with sigmoid

Xavier:  L = Sqrt(6)/Sqrt(Lin+Lout) -----> Good for tanh

He:      Good for relu?

    linear = nn.linear(Lin,Lout) ----------------------------------> Default, L = 1/Sqrt(Lin),            good for sigmoid
    linear.weight.data.uniform_(0, 1) -----------------------------> Uniform, L =1 ,                      not recommended
    nn.init.xavier_uniform_(linear.weight) ------------------------> Xavier,  L = Sqrt(6)/Sqrt(Lin+Lout), good for tanh
    nn.init.kaiming_uniform_(linear.weight,nonlinearity = 'relu') -> He,      unsure,                     good for relu
    

**Gradient Descent with Momentum**

<img src = "https://qph.fs.quoracdn.net/main-qimg-a6be3a9474a5df3012a71e4de21c06c2" width = 600, align = "center">

<img src = "/images/no_mom.png" width = 400, align = "center"> <img src = "/images/mom.png" width = 400, align = "center">

Creates a momentum parameter to help 'keep the ball rolling' 

(ie, it will keep rolling, check other side of the zero gradient section - can tell if its saddle point)

Instead of:
 
     w(k+1) = w(k) - lr x grad(cost(wk))
 
 Do:
 
    v(k+1) = rho x v(k) + grad(cost(wk)) ------> if grad is 0, next v is still non-vanishing to continue checking
    w(k+1) = w(k) - lr x v(k+1)

Good for:
* Saddle Points
* Local Minima
* Gradient Noise
* High Condition Number

PyTorch:

    optimizer = torch.optim.SGD(model.parameters(),lr = 0.1, momentum = 0.4) 
    
    
**Batch Normalization**

Steps:
1. Create a MiniBatch
2. For each layer, normalize the (linear) outputs for that layer according to their batch std and mean.
3. Rescale and shift the normalized outputs.
4. Apply activation function. 

For training, use mean/std for the batch

For prediction, use mean/std for the entire population. (Other methods possible)

Why?
* Rescales contours of loss function to be similiar in all directions
* Minimizes vanishing gradient problem be rescaling to well-behaved range
* 'Reduces Internal Covariate Shift' -> Propagation of small activations early on affecting later layers
* Don't need dropout
* Increase learning rate
* Don't need bias
* **MUCH** Faster convergence (need less epochs)

PyTorch:

    class Net(nn.Module):
    def __init__(self,in_size,n_hidden1,n_hidden2,out_size):
        super(NetBatchNorm,self).__init__()
        self.linear1=nn.Linear(in_size,n_hidden1)
        self.linear2=nn.Linear(n_hidden1,n_hidden2)
        self.linear3=nn.Linear(n_hidden2,out_size)
        self.bn1 = nn.BatchNorm1d(n_hidden1) ---------------------> Extra BatchNorm on the hidden layers 
        self.bn2 = nn.BatchNorm1d(n_hidden2)
    def forward(self,x):
        x=torch.sigmoid(self.bn1(self.linear1(x))) ---------------> Typically do BatchNorm before activation
        x=torch.sigmoid(self.bn2(self.linear2(x))) ---------------> Activation not necessarily sigmoid
        x=self.linear3(x)
        return x
        
    model = Net(in,H1,....,Hn,out)
    model = nn.Sequential(                  -----------------------> Can also be called using Sequential
      nn.Linear(input_dim,hidden_dim),
      nn.BatchNorm1d(hidden_dim),
      nn.Sigmoid(),
      nn.Linear(hidden_dim,hidden_dim),
      nn.BatchNorm1d(hidden_dim),
      nn.Sigmoid(),
      nn.Linear(hidden_dim,output_dim)
        )

**Labs**

Lab 5.1.1 nicely goes over how to include drop out (also introduces new optimizer, torch.optim.Adam())

Lab 5.1.2 does dropout for regression

Lab 5.2.x does different initializations

Lab 5.3.1 does nice momentum example

Lab 5.3.3 really highlights importance of momentum

Lab 5.4.1 shows how much faster batch norm can be

**Extra Resources**

[Dropout: Learning Less To Learn Better](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5)

[Stack Exchange Explaining Dropout Normalization](https://stats.stackexchange.com/questions/241645/how-to-explain-dropout-regularization-in-simple-terms)

[TDS Random Initialization](https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e)

[Medium Random Initialzation](https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94)

[Machine Learning For Physicists (Good Momentum Section)](https://arxiv.org/abs/1803.08823)

[Original Batch Norm Paper](https://arxiv.org/pdf/1502.03167.pdf)

[TDS Batch Normalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)

[What is Covariate Shift](https://recast.ai/blog/internal-covariate-shift/)

[Batch Norm Example](http://rohanvarma.me/Batch-Norm/)
