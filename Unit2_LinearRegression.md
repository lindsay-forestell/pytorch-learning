# Unit 2: Linear Regression

  **Forward Function**

  Important function used by neural net modules

    w = torch.tensor(2.0,requires_grad=True)
    b = torch.tensor('')
    def forward(x):
      return w*x+b
    
 **Linear Class**
 
    from torch.nn import Linear
    model = Linear(in_features=1,out_features=1,bias=True) 
    (randomly sets initial w,b for y = b + wx, number of features +1 from in_features due to bias)
    yhat = model(x) (prediction)
    print(list(model.parameters())) (lists w,b)
  
  **Custom Modules**
   
    import torch.nn as nn
    class LR(nn.module):
      def __init__(self,in,out):
        super(LR,self).__init__() (sets any other necessary parameters from nn.module)
        self.linear = nn.Linear(in,out)
        
       def forward(self,x) 
       (needs to be called 'forward' so that future calls of LR class objects will accurately predict)
        return self.Linear(x)
        
    lr = LR(1,1)
    yhat = lr(x) will predict y as a function of x BECAUSE the forward function is defined in the class
    lr(x), lr.parameters(), etc. from Linear Class all carry over
    
  **Labs**
  
 Lab 2.1.1 contains useful basic modelling info (build custom modules)
 
 **Extra Resources**
 
 [Module Class](https://pytorch.org/docs/stable/nn.html)
  
  
