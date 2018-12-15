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
    
  **Basic Linear Regression**
  
    X  = torch.arange(-3,3,0.1)
    f = f(x) -> y = f + 0.1*torch.randn(X.size()) adds noise to a function
    cost = loss (in pytorch) = (1/N) * sum(y - yhat)^2 = sum(loss) or mean(loss)
   
   **Batch Gradient Descent** = using all data at once when doing single epoch of training
   
    def forward(x):
     return w*x
    def criterion(x):
     return torch.mean((yhat-y)**2)
    LOSS = []
    lr = 0.1
    for epoch in range(niter):
     yhat = forward(x)
     loss = criterion(yhat,y)
     loss.backward() ---------------------------> calculates d(loss)/d(all variables)
     w.data = w.data - lr*w.grad.data ----------> w.data gives tensor(val), w.data.item() gives a #
     w.grad.data.zero_() -----------------------> 
     LOSS.append(loss)
     
   **Stochastic and Mini-Batch Gradient Descent** = using 1/batch of  data instead of all of it when doing single epoch of training
   
    stochastic: X = all data, Y = all data
    for epoch in range(niter):
     Yhat = forward(X); LOSSALL.append(criterion(Yhat,Y)) ---------> can include if you can fit all data on computer
     for (x,y) in zip(X,Y): ----> looks at single x,y element
      yhat, loss, backward, update w,b, LOSS as before
      
   **Using Data Loader** useful way to batch your data, can use for stochastic or mini batch
   
    from torch.utils.data import Dataset
    class Data(Dataset):
     def __init__(self):
      self.x, self.y, self.len = self.x.shape[0] ----------> initialize your data
     def __getitem__(self,index):
       return self.x[index],self.y[index]
      def __len__(self):
       return self.len
       
     data = Data()
     trainloader = Dataloader(dataset = data, batch_size = n) ----> n = 1 for stochastic, 1<m<N mini batch, N batch
     
     replace for (x,y) in zip(X,Y): with --------> for x,y in trainloader:
    
    
  **Labs**
  
 Lab 2.1.1 contains useful basic modelling info (build custom modules)
 
 Lab 2.2.1 contains a useful plotting class for visualizing your model & the cost function - also added stuff about y.backward,x.grad,..
 
 Lab 2.2.2 contains useful plotting class for visualizing 2D cost
 
 Lab 2.3.1 and 2.3.2 contain useful examples of a Dataset loader
 
 **Extra Resources**
 
 [Module Class](https://pytorch.org/docs/stable/nn.html)
  
  
