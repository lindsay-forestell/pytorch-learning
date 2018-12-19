# Unit 5: Computer Vision

**What is Convolution**

Useful for images: helps find patterns that could be in different locations in different images'

Preserves pixel relationships, not absolute pixel locations

<img src = "https://cdn-images-1.medium.com/max/1600/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif" width = 600, align = "center">

Steps:
1. Apply **zero padding** to image so that image and kernel are compatible with **stride**. 
2. Start in top left corner of image. 
3. Overlay **kernel** matrix.
4. Multiply overlaid matrix elements with kernel elements.
5. Sum all elements and add bias. This becomes first element of **activation map**. 
6. Shift kernel over by **stride** columns. Repeat operation.
7. Shift kernel down by **stride** rows. Repeat until finished image. 
8. Apply activation functions. 

Output Length = Floor(1 + (M'-k)/stride)

M' = M + 2 x padding

M = Image Length

k = Kernel Length

Often pad to preserve the original input size

Also note that the right/bottom padding doesn't necessarily need to be used, 

Filter just stops scanning before it would run over the edge

Image Tensor: [number of images, number of channels, image matrix = (a x b)]

PyTorch (Intializes with random weights)

    conv = nn.Conv2d(in_channels = 1 -------------> Greyscale, rgb, etc..
                     out_channels = 1, -----------> Number of filters to apply
                     kernel_size = 3, 
                     stride = 2,
                     padding = 1)
                   
                     
 **Multiple Channels**
 
 If out_channels > 1, simply create more than 1 kernel, and activation map will have multiple layers.  
 
 If in_channels > 1, each kernel will be 3-dimensional, to match the depth of the channels.
 
 Adding more than one kernel allows for searching for more distinct features.
 
 <img src = "https://indoml.files.wordpress.com/2018/03/one-convolution-layer1.png" width = 600, align = "center">
 
 **Activation Map**
 
 Apply an activation function after creating all the new output channels in Z. 
 
    Z = conv(image) ------------> Where image should have 4 dimensions
    A = torch.relu(Z) ----------> Or sigmoid, or tanh... 
    
 **Max Pooling**
 
 For a given kernel size, takes only the maximum value in each region of the kernel. 
 
 Reduces height and width, but leaves number of channels unchanged. 
 
 Usually applied after the convolutional layer.
 
 <img src = "https://cdn-images-1.medium.com/max/1600/1*vbfPq-HvBCkAcZhiSTZybg.png" width = 400, align = "center">
 
 * Reduces number of parameters required
 * Helps with overfitting
 
       max = nn.MaxPool2d(kernel_size = 2, stride = None) ----> stride default = None = kernel_size
       
 **PyTorch CNN**
 
 Example CNN: 
 
 <img src = "/images/CNN_simple.png" width = 700, align = "center">
 
    class CNN(nn.Module):
      def __init__(self,out_1=2,out_2=1):
        super(CNN,self).__init__()
        
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=out_1,kernel_size=2,padding=0) ---------------> CNN 1
        self.relu1=nn.ReLU()
        self.maxpool1=nn.MaxPool2d(kernel_size=2 ,stride=1)
       
        self.cnn2=nn.Conv2d(in_channels=out_1,out_channels=out_2,kernel_size=2,stride=1,padding=0) --> CNN 2 
        self.relu2=nn.ReLU()
        self.maxpool2=nn.MaxPool2d(kernel_size=2 ,stride=1)

        self.fc1=nn.Linear(out_2*7*7,2) --------------------------------------------------------------> Fully Connected
        
    def forward(self,x):
    
        out=self.cnn1(x) -----------------> CNN 1
        out=self.relu1(out) 
        out=self.maxpool1(out)

        out=self.cnn2(out) ---------------> CNN 2
        out=self.relu2(out)
        out=self.maxpool2(out)

        out=out.view(out.size(0),-1) -----> Unroll for final NN
        
        out=self.fc1(out) ----------------> Fully Connected Layer
        return out
    
    model = CNN(out_1,out_2)
    model = nn.Sequential(
                          nn.Conv2d(in_channels=1,out_channels=out_1,kernel_size=2,padding=0),
                          nn.ReLU(),
                          self.maxpool1=nn.MaxPool2d(kernel_size=2 ,stride=1),
                          
                          nn.Conv2d(in_channels=out_1,out_channels=out_2,kernel_size=2,stride=1,padding=0),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=2 ,stride=1),
                          
                          nn.Linear(out_2*7*7,2)
                          )
    
 
 **Useful**
 
 Determing the output size of a convolution, takes as input (height,width) tuple
 
 Kernel size can be rectangular as well
 
    def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1): 
      from math import floor 
      if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
      h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
      w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
      return h, w
      
  Resizing an image
  
    IMAGE_SIZE = 8
    composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed) ------------> Smaller Image
    train_dataset=dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()) -> Regular Image
    
 **Pre-Trained Models**
 
 Using Pre-trained models from Torchvision
 
 Pre-train and then add a single fully connected output layer, only learn last layer
 * Speeds up computation
 * Still get high accuracy
 
 Example steps:
 
     import torchvision.models as models
     
     model = models.resnet18(pretrained = True)
     
     mean = [0.485, 0.456, 0.406]
     std  = [0.229, 0.224, 0.225]
     
     transforms_stuff = transforms.Compose([transforms.Resize(224), ---------------> these are required for resnet18
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])
                                            
     train_data = dataset(root = './data', download = True, transform = transforms_stuff) --> ex dataset could be MNIST
     validation = dataset(root = './data', download = True, transform = transforms_stuff, split = 'test')
     
     for param in model.parameters():
      param.requires_grad = False ------------> Only train your own new layer
      
     model.fc = nn.Linear(512,3) -------------> Input matches wanted output from resnet18, output is whatever you need
     
     optimizer = torch.optim.Adam(
                                  [parameters for parameters in model.parameters() if parameters.requires_grad],
                                  lr = 0.001)
     
     train.... (remembering to use model.train(), model.eval())
                    

**Labs**

Lab 6.1.x goes over basic convolutional models, with nice visualizations.

Lab 6.1.3 does max pooling (not covered in videos) 

Lab 6.2.1 has full CNN network and a useful function for determining filter sizes

Lab 6.2.2 does a CNN on MNIST data

Lab 6.2.3 has a function to resize images

**Extra Resources**

[Intuitive Explanation of Convolutional Nets](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

[Medium: Intuition Conv Nets](https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050)

[TDS: Convolution Nets](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)

[Student Notes on Andrew Ng Course](https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/)

[Torchvision Models](https://pytorch.org/docs/stable/torchvision/models.html)
