import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from matplotlib.colors import ListedColormap

class plot_error_surfaces(object):
    def __init__(self,w_range, b_range,X,Y,n_samples=50,go=True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)
        Z=np.zeros((30,30))
        count1=0
        self.y=Y.numpy()
        self.x=X.numpy()
        for w1,b1 in zip(w,b):
            count2=0
            for w2,b2 in zip(w1,b1):


                yhat= 1 / (1 + np.exp(-1*(w2*self.x+b2)))
                Z[count1,count2]=-1*np.mean(self.y*np.log(yhat+1e-16) +(1-self.y)*np.log(1-yhat+1e-16))
                count2 +=1

            count1 +=1
        self.Z=Z
        self.w=w
        self.b=b
        self.W=[]
        self.B=[]
        self.LOSS=[]
        self.n=0
        if go==True:
            plt.figure()
            plt.figure(figsize=(7.5,5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    def get_stuff(self,model,loss):
        self.n=self.n+1
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
        self.LOSS.append(loss)

    def final_plot(self):
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W,self.B, self.LOSS, c='r', marker='x',s=200,alpha=1)
        plt.figure()
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W,self.B,c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x,self.y,'ro',label="training points")
        plt.plot(self.x,self.W[-1]*self.x+self.B[-1],label="estimated line")
        plt.plot(self.x,1 / (1 + np.exp(-1*(self.W[-1]*self.x+self.B[-1]))),label='sigmoid')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-0.1, 2))
        plt.title('Data Space Iteration: '+str(self.n))
        plt.legend()
        plt.subplot(122)
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W,self.B,c='r', marker='x')
        plt.title('Loss Surface Contour Iteration'+str(self.n) )
        plt.xlabel('w')
        plt.ylabel('b')
        #plt.legend()
        plt.show()

def PlotStuff(X,Y,model,epoch,leg=True):

    plt.plot(X.numpy(),model(X).detach().numpy(),label='epoch '+str(epoch))
    plt.plot(X.numpy(),Y.numpy(),'r')
    if leg==True:
        plt.legend()
    else:
        pass

def testF():
    print(np.ones())

def PlotParameters(model):
    W=model.state_dict() ['linear.weight'].data
    w_min=W.min().item()
    w_max=W.max().item()
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    for i,ax in enumerate(axes.flat):
        if i<10:
             # Set the label for the sub-plot.
            ax.set_xlabel( "class: {0}".format(i))

            # Plot the image.
            ax.imshow(W[i,:].view(28,28), vmin=w_min, vmax=w_max, cmap='seismic')

            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
    plt.show()

def show_data(data_sample):

    plt.imshow(data_sample[0].numpy().reshape(28,28),cmap='gray')
    #print(data_sample[1].item())
    plt.title('y= '+ str(data_sample[1].item()))

def plot_decision_regions_3class(model,data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
    X=data_set.x.numpy()
    y=data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min()-0.1 , X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1 , X[:, 1].max() +0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    XX=torch.torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _,yhat=torch.max(model(XX),1)
    yhat=yhat.numpy().reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:]==0,0],X[y[:]==0,1],'ro',label='y=0')
    plt.plot(X[y[:]==1,0],X[y[:]==1,1],'go',label='y=1')
    plt.plot(X[y[:]==2,0],X[y[:]==2,1],'o',label='y=2')
    plt.title("decision region")
    plt.legend()

class DataLogistic(Dataset):
    def __init__(self):
        self.x=torch.arange(-1,1,0.01).view(-1,1)
        self.y=-torch.zeros(self.x.shape[0],1)
        self.y[self.x[:,0]>0.2]=1

        # Add some noise
        self.random_idx0 = np.random.permutation(np.where(np.logical_and(-0.8<=self.x,self.x<=0.2))[0])[0:8]
        self.random_idx1 = np.random.permutation(np.where(np.logical_and(0.2<self.x,self.x<0.8))[0])[0:5]
        self.y[self.random_idx0]=1
        self.y[self.random_idx1]=0
        self.len=self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

class DataNN(Dataset):
    #  modified from: http://cs231n.github.io/neural-networks-case-study/
    def __init__(self,K=3,N=500):
        D = 2
        X = np.zeros((N*K,D)) # data matrix (each row = single example)
        y = np.zeros(N*K, dtype='uint8') # class labels
        for j in range(K):
          ix = range(N*j,N*(j+1))
          r = np.linspace(0.0,1,N) # radius
          t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
          X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
          y[ix] = j

        self.y=torch.from_numpy(y).type(torch.LongTensor)
        self.x=torch.from_numpy(X).type(torch.FloatTensor)
        self.len=y.shape[0]

    def __getitem__(self,index):

        return self.x[index],self.y[index]
    def __len__(self):
        return self.len
    def plot_stuff(self):
        plt.plot(self.x[self.y[:]==0,0].numpy(),self.x[self.y[:]==0,1].numpy(),'o',label="y=0")
        plt.plot(self.x[self.y[:]==1,0].numpy(),self.x[self.y[:]==1,1].numpy(),'ro',label="y=1")
        plt.plot(self.x[self.y[:]==2,0].numpy(),self.x[self.y[:]==2,1].numpy(),'go',label="y=2")
        plt.legend()
