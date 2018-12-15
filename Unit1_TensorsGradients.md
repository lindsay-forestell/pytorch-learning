# Unit 1: Tensors, Gradients, and Datasets
  
  **Useful functions:**
  
    a = torch.tensor([])
    a.dtype
    a.type()
    a.shape
    a.size()
    a.ndimension()
    a.view(new,shape,you,want)
    tensor = torch.from_numpy(np_array)
    np_array = tensor.numpy()
    tensor = torch.from_numpy(pd.values)
    pd_series = pd.Series(tensor.numpy())
    a[i] = 100
    a[1,1:4] = new_array
    a[1][1:4] does NOT work
    
  **Basic Operations:**
  
    torch.dot(u,v) dot product
    torch.mm(A,B) matrix multiplication
    a.mean()
    a.std()
    a.max()
    a.min()
    np.pi
    torch.sin(a)
    torch.linspace(min,max,nsteps)
    torch.arange(min,max,stepsize)
    plt.plot(x.numpy(),y.numpy())
    
  **Derivatives:**
  
    x = torch.tensor(2.0,requires_grad=True)
    y = fxn(x1,x2) (eg x\**2)
    y.backward()
    x1.grad = dy/dx1(at x1=val)
    
  **Retain graphs:**
  
    x.grad.zero_()
    y.backward(retain_graph=True)
      
  **Derivative over function:**
  
    import torch.nn.functional as F
    x = torch.linspace(size wanted)
    Y = F.relu(x)
    y = torch.sum(F.relu(x))
    y.backward()
      
   **Transform Compose:**
   
    from torchvision import transforms
    dt = transform.Compose([transform1(),transform2(),...])
    
  **Labs:**
  
  Labs 1.1.x and 1.2.x contain useful basic pytorch functions for tensors and .backward() function
  
  Lab 1.3.1 does decent class/datasets/transforms overview
  
  Lab 1.3.2 contains useful show_image function (for MNIST data)
  
  **Extra Resources:**
  
   [Classes](https://docs.python.org/3/tutorial/classes.html)
  
   Pytorch Autograd: 
    
   [Towards Datascience](https://towardsdatascience.com/getting-started-with-pytorch-part-1-understanding-how-automatic-differentiation-works-5008282073ec)
   
   [Mastering Autograd](https://pytorch.org/docs/stable/notes/autograd.html)
   
   [Lecture Notes](http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf)
   
   [Stack Overflow Zero Grad](https://stackoverflow.com/questions/48001598/why-is-zero-grad-needed-for-optimization)
  
