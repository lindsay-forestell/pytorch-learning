# Unit 3: Logistic and Softmax Regression

**Logistic Regression**

For classifications: y = 0 or 1

P(y = 1) = sigmoid(z) = 1/(1+exp(-z))

z = xw + b

<img src = "https://ibm.box.com/shared/static/1rpau4ggzepzxzu01p2j4506d5kvobbj.png" width = 600, align = "center">

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import nn, optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.datasets as dsets
    
3 methods to build sigmoid function

    sig = nn.Sigmoid()
    yhat = sig(z)
    yhat = torch.sigmoid(z)
    yhat = F.sigmoid(z) ---------> deprecated now
    
2 methods to build model

    class logistic_regression(nn.Module):
      def __init__(self,n_inputs):
        super(logistic_regression,self).__init__()
        self.linear=nn.Linear(n_inputs,1)
      def forward(self,x):
        yhat=torch.sigmoid(self.linear(x))
        return yhat
        
    model1 = nn.Sequential(nn.Linear(in_features,1),nn.Sigmoid())    
    model2 = logistic_regression(in_features)
    
    x = torch.tensor([[----- sample 1 ------],[----- sample 2 -----],.....])
    yhat = model(x)
    
 Once model is built, can build and train same as with linear regression

 (data -> trainloader, model, criterion, optimizer...)
 
 Determine Accuracy:
 
    yhat = model(x)
    label = yhat > 0.5 ----------------------------------------------------------> Boolean ByteTensor (integer)
    accuracy = torch.mean(label == y.type(torch.ByteTensor)).type(torch.float) --> Must be same types of tensors
    
 Loss Functions - Typically want to use Binary Cross Entropy Loss
 
    criterion_rms = nn.MSELoss() --> Performs poorly with Logistic Regression: Many flat regions, parameters don't update
    criterion_bce = nn.BCELoss() --> Based on maximizing the log likelihood function for a Bernoulli distributed variable
    criterion_ce = nn.CrossEntropyLoss() --> Same as BCE but allows for more than one output, DON'T need to include a sigmoid
    
    e.g of (Binary) Cross Entropy Loss:
    def criterion(yhat,y):
      return -1*torch.mean(y*torch.log(yhat) + (1-y)*torch.log(1-yhat))
    
**SoftMax Regression**

For multi-class classification

PyTorch is built so that the CrossEntropyLoss() function automatically converts the layer to probabilities:

P(y=1|zi,x) = e^zi/(sum(e^zi)) 

As such, don't need to include the probability sigmoid function to the linear model

Correct class corresponds to the zi with the largest value (ie which weight vector points in closest direction to input)

(Which by construction corresponds to the largest hidden probability in the CEL function)


<img src = "/images/softmax1.png" width = 300, align = "center">
<img src = "/images/softmax2.png" width = 300, align = "center">
<img src = "/images/softmax3.png" width = 300, align = "center">

    class SoftMax(nn.Module): -----------------------> This is technically just the linear model again
      def __init__(self,in,out):
        Super(Softmax,self).__init__()
        self.linear = nn.Linear(in,out)
      def forward(self,x):
        return self.linear(x)
        
    model1 = Softmax(in,out)
    model2 = nn.Sequential(nn.Linear(in,out)) -------> Either way works
    
    criterion = nn.CrossEntropyLoss()
    z = model(x)
    _, yhat = z.max(1) ------------------------------> Index of maximum value wrt axis 1 = column, gives the predicted class
    loss = criterion(z,y.type(torch.LongTensor)) ----------> requires that y be of type torch.long
                                                 ----------> z will have multiple columns, loss function auto takes care of it
    
Validating:
    
    for x_test,y_test in validation_loader:
      z = model(x_test)
      _, yhat = torch.max(z.data,1)
      correct += (yhat == y_test).sum().item())
     
    accuracy=correct/N_test
    
**Unrolling a 2D Input**

Need to have feature vectors as single rows, images usually come as m\*n matrices

    x = torch.tensor of size (number of examples, m, n)
    z = model(x.view(-1,m*n)) will be unrolled into 2D tensor
    
    
**Labs**

Lab 3.2.3 contains useful logistic training implementation

Lab 3.3.2 contains useful parameter visualization plotting functions and a SoftMax example

**Extra Resources**

[nn.Sequential](https://pytorch.org/docs/stable/nn.html#sequential)

[How to write a PyTorch Sequential Model](https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model)

[Cross Entropy Loss](https://pytorch.org/docs/stable/nn.html#crossentropyloss)
    
    
