import torch
import numpy as np


class net_basic(torch.nn.Module):
    def __init__(
            self, in_features=1, out_features=1, width=100, bias_flag=False, mode='Vanilla',
            one_d_flag=False,
            green_activation_for_pcnn_2d = False,
    ):
        super(net_basic, self).__init__()
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.one_d_flag =one_d_flag
        self.width = width
        self.bias_flag = bias_flag
        self.nns = torch.nn.ModuleList([
            torch.nn.Linear(in_features=self.in_features, out_features=self.width, bias=True),
            # torch.nn.Linear(in_features=self.width, out_features=self.width, bias=True),
            # torch.nn.Linear(in_features=self.width, out_features=self.width, bias=True),
            # torch.nn.Linear(in_features=self.width, out_features=self.width, bias=True),
            torch.nn.Linear(in_features=self.width, out_features=self.out_features, bias=False),
        ])
        # self.nns[1].requires_grad_(False)
        r'''
            the $\nu$ in Eq. (13) is set to be optimized to realize the $\alpha$ in Eq. (14)
        '''
        self.nu = torch.nn.Parameter(torch.ones(1)[0], requires_grad=True)

        #
        '''
           rand is regard to the uniform distribution U[0, 1]
           and randn is normal N[0, 1]

           I think we are going to use the randn here to generate the gaussiance distributed $v_k$ ?
        '''
        self.w_k = torch.nn.Parameter(
            torch.rand(size=[self.in_features, self.width]) / np.sqrt(self.in_features), requires_grad=True)
        self.a_k = torch.nn.Parameter(
            torch.rand(size=[self.width]) / np.sqrt(self.in_features), requires_grad=True)
        self.v_k = torch.nn.Parameter(
            torch.rand(size=[self.width, 1]) / np.sqrt(self.width), requires_grad=True)
        # self.v_k = torch.randn(size=[self.width, 1], requires_grad=False)
        if mode == 'Vanilla':
            self.activation = torch.nn.ReLU()
        elif mode == 'Physics-informed':
            self.activation = torch.nn.Tanh()

        elif mode == 'Physics-constrained':
            # self.activation = self.cos_activaton
            self.activation = self.bessel0_activation if not self.one_d_flag else self.sin_activaton
            if green_activation_for_pcnn_2d:
                print('\n\n' + '-'*60 + '\n' + 'Green function used as the activation!')
                self.activation = self.green_activation
        # self.activation_other = torch.nn.Tanh()
        else:
            raise ValueError

    def forward(self, x):
        '''

        :param x: in shape of (num_samples, in_features)
        :return:
        '''
        # y = self.activation(self.nns[0](x)) @ self.v_k
        # y = self.nns[2](self.activation(self.nns[1](self.activation(self.nns[0](x)))))
        # y = self.nns[1](self.activation(self.nns[0](x)))
        # temp = x @ self.w_k
        # n = len(self.nns)
        # for i in range(n - 1):
        #     x = self.activation(self.nns[i](x))
        # y = self.nns[n - 1](x)
        y = self.activation(x@self.w_k + self.a_k) @ self.v_k
        return y

    def sin_activaton(self, x):
        # return self.activation_other(x)
        return torch.sin(x)

    def bessel0_activation(self, x):
        t = torch.ones_like(x)
        s_um = torch.ones_like(x)
        x_half_2 = ( x / 2)**2
        for i in range(1, 40):
            t *= -x_half_2/i**2
            s_um += t
        # s_um = torch.special.bessel_j0(x)
        return s_um

    def green_activation(self, x):
        h_real = torch.cos(x)/x
        h_imagin = torch.sin(x)/x  # there is no where to times the
                                   # imaginary part together as this is a single hidden layer NN
        return h_real

    def forward_ddy(self, x):
        '''
                This is used to calculated the ddy and the
        :param x:
        :return:
        '''
        # x.requires_grad = True
        g = x.clone()
        g.requires_grad = True
        y = self.forward(x)

        # x = torch.tensor([1.*np.pi/6., 2.*np.pi/6., 3.*np.pi/6., 4.*np.pi/6.], requires_grad=True)

        def y_pre(xx):
            return self.forward(xx).sum()
            # return torch.sin(xx).sum()

        def dy_pre(xx):
            return torch.autograd.functional.jacobian(y_pre, xx, create_graph=True).sum()

        # dy_test = torch.autograd.functional.jacobian(y_pre, g)

        ddy = torch.autograd.functional.jacobian(dy_pre, g)

        return y, ddy
