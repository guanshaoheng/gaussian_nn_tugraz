# -*- coding: utf-8 -*-
"""
Created on Fri May 12 19:28:57 2022

@author: sranf
"""

%reset -f

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.autograd as autograd

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import imageio

import time

starttime = time.time()




import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.close('all')

seed = 1
torch.manual_seed(seed)    # reproducible
Ndata =11
Ntest = 22
Niter_vanilla = 10**4 #100000
Niter_helm =    10**4 #100000
Niter_pinn =    10**4# 100000
lr_vanilla = .001
lr_helm = .001
lr_pinn = .001


lamda = 1
lamda_pinn = .955
noise_level = 0.2
omega = .51
phi=0.50001
scale = 2*np.pi
a_1 = 2*omega
k=1
LW = 2

def truth(x):
    y = np.sin(2*omega*x + phi)     
    return y

x = torch.unsqueeze(torch.linspace(0, scale, Ndata), dim=1)  # x data (tensor), shape=(100, 1)
y = truth(x) + noise_level*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

xdata = x
ydata = y



# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

xtest = torch.unsqueeze(torch.linspace(0, 2*scale, Ntest), dim=1)
xtestval = torch.unsqueeze(torch.linspace(float(x[-1])+2*scale/Ndata, 2*scale, Ndata), dim=1)
xplot = torch.unsqueeze(torch.linspace(0, 2*scale, 1000), dim=1)

#xtest = torch.cat((xtest, xtest2),0)

# this is one way to define a network
class Net_vanilla(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_vanilla, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.loss_function = torch.nn.MSELoss(reduction='mean')
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
    
class Net_helmholtz(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_helmholtz, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.loss_function = torch.nn.MSELoss(reduction='mean')
        
    def forward(self, x):
        x = torch.sin(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x    
    
    def loss_PDE(self, x_to_train_f):
                
        x_1_f = x_to_train_f[:,[0]]

                        
        g = x_to_train_f.clone()
                        
        g.requires_grad = True
        
        u = self.forward(g)
                
        u_x = autograd.grad(u,g,torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
                                
        u_xx = autograd.grad(u_x,g,torch.ones(x_to_train_f.shape).to(device), create_graph=True)[0]
                                                            
        u_xx_1 = u_xx[:,[0]]
        
   
                
        #q = ( -(a_1*np.pi)**2    + k**2 ) * torch.sin(a_1*np.pi*x_1_f)  
                        
        f = u_xx_1   + k**2 * u 
        # print("a = ", u_xx)
        # print("b = ", u_x)
        # print("c = ", u)
        X_f_train = x
        f_hat = torch.zeros(X_f_train.shape[0],1).to(device)
        loss_f = self.loss_function(f,f_hat)
    
        
        return loss_f

class Net_pinn(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_pinn, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.loss_function = torch.nn.MSELoss(reduction='mean')
        
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
   
    def loss_PDE(self, x_to_train_f):
                
        x_1_f = x_to_train_f[:,[0]]

                        
        g = x_to_train_f.clone()
                        
        g.requires_grad = True
        
        u = self.forward(g)
                
        u_x = autograd.grad(u,g,torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
                                
        u_xx = autograd.grad(u_x,g,torch.ones(x_to_train_f.shape).to(device), create_graph=True)[0]
                                                            
        u_xx_1 = u_xx[:,[0]]
        
   
                
        q = ( -(a_1*np.pi)**2    + k**2 ) * torch.sin(a_1*np.pi*x_1_f)  
                        
        f = u_xx_1   + k**2 * u  
        # print("a = ", u_xx)
        # print("b = ", u_x)
        # print("c = ", u)
        X_f_train = x
        f_hat = torch.zeros(X_f_train.shape[0],1).to(device)
        loss_f = self.loss_function(f,f_hat)
    
        
        return loss_f
    


######### VANILLA EXPERIMENT and PLOT
######### VANILLA EXPERIMENT and PLOT
######### VANILLA EXPERIMENT and PLOT
######### VANILLA EXPERIMENT and PLOT
Niter = Niter_vanilla
net = Net_vanilla(n_feature=1, n_hidden=10, n_output=1)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters(), lr_vanilla)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
loss_vector_vanilla = np.zeros(Niter)
test_loss_vanilla = np.zeros(Niter)
# train the network
for t in range(Niter):
  
    prediction = net(x)     # input x and predict based on x 

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    loss_vector_vanilla[t] = loss
    test_loss_vanilla[t] = loss_func(net(xtestval),truth(xtestval))
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients


prediction_plot_vanilla = net(xplot)

plt.figure(1)
plt.plot(xplot,truth(xplot), 'k-', label = 'truth')
plt.scatter(x.data.numpy(), y.data.numpy(), color = "black")
plt.plot(xplot.data.numpy(), prediction_plot_vanilla.data.numpy(), 'g-.', lw=LW)

plt.figure(2)
plt.semilogy(loss_vector_vanilla, 'g-', lw=LW, label = 'vanilla')
plt.semilogy(test_loss_vanilla, 'g--', lw=LW)

plt.figure(3)
plt.loglog(loss_vector_vanilla, 'g-', lw=LW, label = 'vanilla')
plt.loglog(test_loss_vanilla, 'g--', lw=LW)




######### HELMHOLTZ EXPERIMENT and PLOT
######### HELMHOLTZ EXPERIMENT and PLOT
######### HELMHOLTZ EXPERIMENT and PLOT
######### HELMHOLTZ EXPERIMENT and PLOT

Niter = Niter_helm
net = Net_helmholtz(n_feature=1, n_hidden=10, n_output=1)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters(), lr_helm)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
loss_vector_helmholtz = np.zeros(Niter)
test_loss_helmholtz = np.zeros(Niter)
# train the network
for t in range(Niter):
    

    prediction = net(x)     # input x and predict based on x 
    loss = loss_func(prediction, y)  + lamda* Net_helmholtz.loss_PDE(net, x)   + lamda* Net_helmholtz.loss_PDE(net, xtestval)  # must be (1. nn output, 2. target)
    loss_vector_helmholtz[t] = loss_func(prediction, y)
    test_loss_helmholtz[t] = loss_func(net(xtestval),truth(xtestval))
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradient

prediction_plot_helm = net(xplot)

# plot and show learning process
plt.figure(1)
plt.plot(xplot.data.numpy(), prediction_plot_helm.data.numpy(), 'r-', lw=LW)

plt.figure(2)
plt.semilogy(loss_vector_helmholtz, 'r-', lw=LW, label = 'physics-consistent')
plt.semilogy(test_loss_helmholtz, 'r--', lw=LW)

plt.figure(3)
plt.loglog(loss_vector_helmholtz, 'r-', lw=LW, label = 'physics-consistent')
plt.loglog(test_loss_helmholtz, 'r--', lw=LW)

######### PINN EXPERIMENT AND PLOT
######### PINN EXPERIMENT AND PLOT
######### PINN EXPERIMENT AND PLOT
######### PINN EXPERIMENT AND PLOT
#f_hat = torch.zeros(X_f_train.shape[0],1).to(device)

Niter = Niter_pinn
net = Net_pinn(n_feature=1, n_hidden=10, n_output=1)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters(), lr_pinn)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
loss_vector_pinn = np.zeros(Niter)
test_loss_pinn = np.zeros(Niter)
# train the network
for t in range(Niter):
  
    prediction = net(x)     # input x and predict based on x 
    pred_pivots = net(xtest)
    loss = loss_func(prediction, y)   + lamda_pinn* Net_pinn.loss_PDE(net, x) + lamda_pinn* Net_pinn.loss_PDE(net, xtestval)      # must be (1. nn output, 2. target)
    loss_vector_pinn[t] = loss_func(prediction, y) 
    test_loss_pinn[t] = loss_func(net(xtestval),truth(xtestval))
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradient

prediction_plot_pinn = net(xplot)

# plot and show learning process
plt.figure(1) 
plt.plot(xplot.data.numpy(), prediction_plot_pinn.data.numpy(), 'b:', lw=LW)


plt.xlabel('x', fontsize =14)
plt.ylabel('f(x)', fontsize =14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tick_params('both', length=6, width=2, which='major',direction="in")
plt.tick_params('both', length=3, width=1, which='minor',direction="in")

################################################



plt.figure(2)
plt.semilogy(loss_vector_pinn, 'b-', lw=LW, label = 'physics-informed')
plt.semilogy(test_loss_pinn, 'b--', lw=LW)
plt.xlabel('iteration', fontsize =14)
plt.ylabel('error', fontsize =14)

plt.ylim(1*10**-5,100)
plt.xlim(0,np.max([Niter_vanilla,Niter_helm,Niter_pinn]))
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize =14, bbox_to_anchor=(.805, 0.51, 0.18, 0.5)) 

plt.savefig('convergence_plot_ADAM.png')

plt.figure(3)
plt.loglog(loss_vector_pinn, 'b-', lw=LW, label = 'physics-informed')
plt.loglog(test_loss_pinn, 'b--', lw=LW)

plt.xlabel('iteration', fontsize =14)
plt.ylabel('error', fontsize =14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tick_params('both', length=6, width=2, which='major',direction="in")
plt.tick_params('both', length=3, width=1, which='minor',direction="in")





#add legend to plot

plt.figure(3)
plt.ylim(1*10**-5,100)
plt.xlim(0,np.max([Niter_vanilla,Niter_helm,Niter_pinn]))
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize =14, bbox_to_anchor=(.5, 0.51, 0.18, 0.5)) 
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize =14, bbox_to_anchor=(.1, 0.1, 0.18, 0.5))

handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [0,2,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize =14, bbox_to_anchor=(.399, -0.099, 0.18, 0.5)) 

plt.savefig('convergence_plot_ADAM_log.png')
#plt.legend(loc='lower right', fontsize =14, bbox_to_anchor=(.5, 0.1, 0.5, 0.5))

plt.figure(2)
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize =14, bbox_to_anchor=(.83, 0.35099, 0.18, 0.5)) 

plt.savefig('convergence_plot_ADAM.png')


plt.figure(1)
x = torch.unsqueeze(torch.linspace(0, 2*np.pi, Ndata*100), dim=1)  
y = np.sin(2*omega*xtest + phi)   
x, y = Variable(x), Variable(y)
plt.plot(xdata, ydata, 'ko', label='data',  markerfacecolor='white')
plt.plot(xtestval, truth(xtestval), 'kv',  markerfacecolor='black', label='test pivots')
plt.legend()

plt.ylim(-1.1,2)
plt.xlim(-.1,2*2*np.pi+0.1)

plt.savefig('solution_plot_ADAM.png')

winsound.Beep(frequency, duration)

endtime = time.time()
print(endtime - starttime)