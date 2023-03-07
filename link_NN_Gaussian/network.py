import torch
import numpy as np


class net_basic(torch.nn.Module):
    def __init__(self, in_features=1, out_features=1, width=100, bias_flag=False, mode='vallina'):
        super(net_basic, self).__init__()
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.bias_flag = bias_flag
        self.nns = torch.nn.ModuleList([
            torch.nn.Linear(in_features=self.in_features, out_features=self.width, bias=True),
            torch.nn.Linear(in_features=self.width, out_features=self.out_features, bias=True)
        ])
        # self.nns[1].requires_grad_(False)
        r'''
            the $\nu$ in Eq. (13) is set to be optimized to realize the $\alpha$ in Eq. (14)
        '''
        self.nu = torch.nn.Parameter(torch.ones(1)[0], requires_grad=False)
        # TODO randn or rand
        '''
           rand is regard to the uniform distribution U[0, 1]
           and randn is normal N[0, 1]
           
           I think we are going to use the randn here to generate the gaussiance distributed $v_k$ ?
        '''
        self.w_k = torch.nn.Parameter(torch.randn(size=[1, self.width])/np.sqrt(self.width), requires_grad=True)
        self.a_k = torch.nn.Parameter(torch.randn(size=[self.width])/np.sqrt(self.width), requires_grad=True)
        self.v_k = torch.nn.Parameter(torch.randn(size=[self.width, 1])/np.sqrt(self.width), requires_grad=False)
        # self.v_k = torch.randn(size=[self.width, 1], requires_grad=False)
        if mode != 'physics_constrained':
            # self.activation = torch.nn.ReLU()
            # self.activation = torch.nn.Sigmoid()
            self.activation = torch.nn.Tanh()
        else:
            self.activation = self.sin_activaton
        # self.activation_other = torch.nn.Tanh()

    def forward(self, x):
        '''

        :param x: in shape of (num_samples, in_features)
        :return:
        '''
        # y = self.activation(self.nns[0](x)) @ self.v_k
        # y = self.nns[2](self.activation(self.nns[1](self.activation(self.nns[0](x)))))
        # y = self.nns[1](self.activation(self.nns[0](x)))
        # temp = x @ self.w_k
        y = self.activation(x@self.w_k + self.a_k) @ self.v_k
        return y

    def sin_activaton(self, x):
        # return self.activation_other(x)
        return torch.sin_(x)

    def forward_ddy(self, x):
        '''
                This is used to calculated the ddy and the
        :param x:
        :return:
        '''
        # x.requires_grad = True
        g = x.clone()
        g.requires_grad = True
        y = self.forward(g)
        dy = torch.autograd.grad(y, g, grad_outputs=torch.ones_like(g), create_graph=True, retain_graph=True)[0]
        ddy = torch.autograd.grad(dy, g, grad_outputs=torch.ones_like(g),  create_graph=True, retain_graph=True)[0]
        '''
        '''
        temp = ddy + self.nu**2*y
        return y, temp
