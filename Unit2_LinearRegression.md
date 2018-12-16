# Unit 2: Linear Regression

 **Important Classes**
 
  class Data(Dataset) ---------------------> must contain getitem and len functions
  class model_type(nn.Module) -------------> must contain forward function

  **Forward Function**

  Important function used by neural net modules

    w = torch.tensor(2.0,requires_grad=True)
    b = torch.tensor('')
    def forward(x):
      return w*x+b
    
 **Linear Class**
 
    from torch.nn import Linear
    model = Linear(in_features=1,out_features=1,bias=True) --------> n features = in_features+1 due to bias
                                                           --------> randomly sets w,b
    model.state_dict()['linear.weight'][0] = -15 ------------------> can change the random init by hand if you want
    model.state_dict()['linear.bias'][0] = -10
    
    yhat = model(x) -----------------------------------------------> prediction
    print(list(model.parameters())) -------------------------------> lists w,b
  
  **Custom Modules**
   
    import torch.nn as nn
    class LR(nn.module):
      def __init__(self,in,out):
        super(LR,self).__init__() --> sets any other necessary parameters from nn.module
        self.linear = nn.Linear(in,out)
        
       def forward(self,x) ---------> needs to be called 'forward' so that future calls of LR class objects will accurately predict
        return self.Linear(x)
        
    lr = LR(1,1)
    yhat = lr(x) -------------------> will predict y as a function of x BECAUSE the forward function is defined in the class
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
     w.grad.data.zero_() -----------------------> always need to re-zero the gradient
     LOSS.append(loss)
     
   **Stochastic and Mini-Batch Gradient Descent** = using 1/batch of data instead of all of it when doing single epoch of training
   
    stochastic: X = all data, Y = all data
    for epoch in range(niter):
     Yhat = forward(X); LOSSALL.append(criterion(Yhat,Y)) ---------> can include if you can fit all data on computer
     for (x,y) in zip(X,Y): ---------------------------------------> looks at single x,y element
      yhat, loss, backward, update w,b, LOSS as before
      
   **Using Data Loader** useful way to batch your data, can use for stochastic or mini batch
   
    from torch.utils.data import Dataset
    class Data(Dataset):
     def __init__(self):
      self.x, self.y, self.len = self.x.shape[0] ----------> initialize your data
     def __getitem__(self,index): -------------------------> always need to override getitem and len functions
       return self.x[index],self.y[index]
      def __len__(self):
       return self.len
       
     data = Data()
     trainloader = Dataloader(dataset = data, batch_size = n) ----> n = 1 for stochastic, 1<m<N mini batch, N batch
     
     replace for (x,y) in zip(X,Y): with --------> for x,y in trainloader:
    
   **Full Blown PyTorch Method**
   
   <img src = "https://ibm.box.com/shared/static/oorp4w5ucfahk3vuf333nt9zrw9hefek.png" width="400" alt="Model Cost with optimizer" />
   
    from torch import nn, optim
    from torch.utils.data import Dataset, DataLoader
   
    dataset = Data() ----------------------------------------------> based on whatever Data model you have defined
    model = LR(in_params,out_params) ------------------------------> automatically has in+1 input params, 1 is bias
    criterion = nn.MSELoss() --------------------------------------> built in loss function
    trainloader = DataLoader(dataset = dataset, batch_size = n) ---> choose batch size
    optimizer = optim.SGD(model.parameters(),lr=0.01) -------------> requires an iterable parameter set as input
                                                                     note that SGD algorithm is just vanilla GD
    
    def train(model,criterion,trainloader,optimizer,epochs):
     for epoch in range(epochs):
      for x,y in trainloader:
       yhat = model(x)
       loss = criterion(yhat,y)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step() --------> automatically performs the optimization steps on all parameters & updates them
       
  **Training and Validation**
  
  Split up data - no real way recommended
  
  Eg. train = Data(), validate = Data(train=False)
  
  Make hyperparameter list: eg learning_rates = [0.001,0.01,0.1,1]
  
  Loop over hyperparameters, choose the one that minimizes *loss(VALIDATION DATA)
  
    for i,lr in enumerate(learning_rates):
     model = ... optimizer = ...
     TRAIN MODEL
     save training loss, validation loss, model
     
  **Early Stopping** 
  
  Note that this has different meanings in literature sometimes
  
  Here taken to mean data has large outliers, so best model may not be largest epoch
  
    min_loss = 100000000 ---------------------------------------> some big number
    for epoch in range(epochs):
     for x,y in trainloader:
      TRAIN, determine val_loss, model
      if val_loss < min_loss: ----------------------------------> always use validation loss here 
        value = epoch
        min_loss = loss_val
        torch.save(model.state_dict(), 'best_model.pt') --------> SAVES BEST MODEL, even if it's not the last epoch
    
    model_best = LR(1,1) ---------------------------------------> load some random model
    model_best.load_state_dict(torch.load('best_model.pt')) ----> then give it the best fit parameters
    
  **Multiple Parameters**
  
   <img src = "https://ibm.box.com/shared/static/768cul6pj8hc93uh9ujpajihnp8xdukx.png" width = 600, align = "center">
  
    X = torch.tensor([[---x1---],[---x2---],...]) ------------> each ROW xi = example i = [x1,x2,...] with n features
    w = torch.tensor([[w1],[w2],...,[wn],requires_grad=True) -> w = column vector with n features
    b = torch.tensor(b,requires_grad = True) ------------------> b = scalar
    
    def forward(x):
     return torch.mm(X,w) + b ---------------------------------> use matrix multiplication to get predictions
    
    LR(in,out) ------------------------------------------------> call with in = number of x features
               ------------------------------------------------> out = number y features ( total # params = (in+1)*out )
    
    model.parameters() ----------------------------------------> to see the parameters in the model
    model.state_dict()
    
   Everything else is the same as single variable, just changing LR(in,out) to in > 1, out > 1
    
  **Labs**
  
 Lab 2.1.1 contains useful basic modelling info (build custom modules)
 
 Lab 2.2.1 contains a useful plotting class for visualizing your model & the cost function - also added stuff about y.backward,x.grad,..
 
 Lab 2.2.2 contains useful plotting class for visualizing 2D cost
 
 Lab 2.3.1 and 2.3.2 contain useful examples of a Dataset loader
 
 Lab 2.4 contains useful full example of linear regression the pytorch way
 
 Lab 2.5.2 contains useful loading and saving in torch info
 
 Lab 2.6.2 contains useful 2D plotting function
 
 **Extra Resources**
 
 [Module Class](https://pytorch.org/docs/stable/nn.html)
 
 [Datasets](https://pytorch.org/docs/stable/data.html)
 
 [Extra Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
 
 [Extra Optimizers](https://pytorch.org/docs/stable/optim.html#algorithms)
 
 [Explain SGD Optimizer](https://discuss.pytorch.org/t/how-sgd-works-in-pytorch/8060/9)
  
 Early Stopping (As it often has a different definition: stop training when some parameter stops improving:
 
 [When? Paper](http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)
 
 [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
 
 [Example Pytorch Code](https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d)
  
