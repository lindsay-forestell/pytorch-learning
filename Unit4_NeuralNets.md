# Unit 4: Feed Forward Neural Networks

**Basic Neural Network**

<img src = "/images/nn1.png" width = 600, align = "center">

    import torch 
    import torch.nn as nn
    
    class Net(nn.Module): ------------------------> Basic Single Layer NN Class
      def __init__(self,D_in,H,D_out): -----------> Requires number of input, hidden, and output nodes
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.linear2=nn.Linear(H,D_out)
      def forward(self,x):
        a1 = torch.sigmoid(self.linear1(x))
        out = torch.sigmoid(self.linear2(a1)) -----> Leave off the sigmoid if doing softmax for multiclass
        return out
        
    model = Net(in,H,out)
    model = nn.Sequential(nn.Linear(in,H), ---> 2nd way to build a NN model
                          nn.Sigmoid(),    ---> Note that x goes into input 1, which becomes input into 2, etc. 
                          nn.Linear(H,out), --> Also note that classes fed into Sequential should be things that
                          nn.Sigmoid())     --> set up the model, not things that take as input the data itself 
                                            --> ie nn.Sigmoid() not torch.sigmoid(x) 
                                            --> ** Must be Module subclasses ** 
                                            
    criterion = nn.BCELoss() ------------------> As sigmoid is included in last layer already, need to use BCE loss
    criterion = nn.CrossEntropyLoss() ---------> If we leave Sigmoid off last layer, use this. Multiple categories. 
                                      ---------> Can leave Sigmoid on last layer if you want, but not great practice. 
                                      ---------> If we want to use this for reg. binary class, D_out = 2 instead of 1
                                      ---------> compares yhat = [[value1, value2, value3....]] to y = [class index]
                                      ---------> y must be long & 1 dimensional: y.type(torch.LongTensor).view(-1)
    criterion = nn.MSELoss() ------------------> Useful for regression problems, also leave sigmoid off last layer
    
**Overfitting**

<img src = "/images/nn2.png" width = 400, align = "center">

<img src = "/images/nn3.png" width = 400, align = "center">

Avoid overfitting by using validation set to determine number of hidden nodes to use. 

Can also use regularization. 

**Backpropagation**

Reduced required number of computations by reusing derivatives (chain rule)

Caution: sigmoid has vanishing gradient problem, so often good to use different activation functions

**Activation Functions**

<img src = "/images/activation.png" width = 300, align = "center"> <img src = "/images/activation_deriv.png" width = 300, align = "center">

Tanh somtimes outperforms ReLU for small epochs (in terms of accuracy), but ReLU typically takes over in the end

ReLU also outperforms for large (>~ 10) numbers of hidden layers

    def forward(self,x):
        a1 = torch.sigmoid(self.linear1(x)) OR 
             torch.tanh(self.linear1(x)) OR
             torch.relu(self.linear1(x))
        yhat = self.linear2(x)
        return yhat
    
    nn.Sequential(nn.Linear(in,H), 
                  nn.Sigmoid() OR 
                  nn.Tanh() OR 
                  nn.ReLU(),
                  nn.linear(H,out))
                  
**Deep Networks**

More layers typically raises the accuracy on the validation set

    class Net(nn.Module):
      def __init__(self,D_in,H1,H2,...,HN,D_out): ---------> Requires number of input, hidden, and output nodes
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H1) -------------------> One linear map for every pair of layers
        self.linear2=nn.Linear(H1,H2)
        ...
        self.linear(N+1) = nn.Linear(HN,D_out)
        
      def forward(self,x):
        x = torch.activation(self.linear1(x)) -------------> One activation for every pair of layers
        x = torch.activation(self.linear2(x)) -------------> activation = sigmoid, tanh, relu
        ...
        x = torch.activation(self.linearN(x))
        x = self.linear(N+1)(x) ----------------------------> Leave off the activation if doing softmax for multiclass
        return x
        
      model = Net(D_in,H1,H2,...,HN,D_out)
      model = nn.Sequential(nn.Linear(D_in,H1),   nn.Activation(),
                            nn.Linear(H1,H2),     nn.Activation(),
                            ...
                            nn.Linear(H(N-1),HN), nn.Activation(),
                            nn.Linear(HN,D_out))
                            
 For a variable number of layers:
 
    class Net(nn.Module):
      def __init__(self,Layers): ----------------> Input is a list containing layer sizes: [D_in,H1,H2,....,HN,D_out]
        super(Net,self).__init__()
        self.hidden = nn.ModuleList()

        for input_size,output_size in zip(Layers,Layers[1:]): ---> Creates all the linear layers
            self.hidden.append(nn.Linear(input_size,output_size))
        
      def forward(self,activation):
        L=len(self.hidden)
        for (l,linear_transform)  in zip(range(L),self.hidden): ---> Creates all the activation layers
            if l<L-1:
                activation =torch.relu(linear_transform(activation)) 
           
            else:                                               ----> Leaves the last layer linear
                activation =linear_transform(activation)
        return activation

**Useful**

Print out your parameters:

    def print_model_parameters(model):
      for i in range(len(list(model.parameters()))):
        if (i+1)%2==0:
            print("the number of bias parameters for layer",i)
        else:
            print("the number of parameters for layer",i+1)
    
        print( list(model.parameters())[i].size() )
        
 Unroll images into usuable rows of features
 
    z = model(x.view(-1,28*28))

**Labs**

Lab 4.1.1 contains a good simple example of a NN

Lab 4.1.2 contains good visualizations for the non-linearities encapsulated by NN, and shows both BCE and CE on same data set

Lab 4.1.3 contains a good function for printing all the parameters

Lab 4.4.1 (for example) contains a good example of a training function that stores accuracy and losses

Lab 4.4.2 has good example of full blown NN with variable hidden layer sizes and depth

**Extra Resources**

[When to use ModuleList vs Sequential](https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463)

[Other Optimizers](https://pytorch.org/docs/stable/optim.html)

[How to add L2 Reg](https://discuss.pytorch.org/t/how-to-add-a-l2-regularization-term-in-my-loss-function/17411/8)

**Questions**

How to regularize?

Include weight_decay = lambda in the optimizer (automatically leaves it out of the cost, but still accounts for it?)
