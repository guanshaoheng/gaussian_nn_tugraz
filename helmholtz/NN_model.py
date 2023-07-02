import numpy as np
import torch


class net_basic(torch.nn.Module):
    def __init__(
            self, in_features=1, out_features=1, width=100, bias_flag=False, mode='Vanilla',
            one_d_flag=False,
            green_activation_for_pcnn_2d=False,
    ):
        super(net_basic, self).__init__()
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.one_d_flag = one_d_flag
        self.width = width
        self.bias_flag = bias_flag
        # self.nns = torch.nn.ModuleList([
        #     torch.nn.Linear(in_features=self.in_features, out_features=self.width, bias=True),
        #     # torch.nn.Linear(in_features=self.width, out_features=self.width, bias=True),
        #     # torch.nn.Linear(in_features=self.width, out_features=self.width, bias=True),
        #     # torch.nn.Linear(in_features=self.width, out_features=self.width, bias=True),
        #     torch.nn.Linear(in_features=self.width, out_features=self.out_features, bias=False),
        # ])
        # self.nns[1].requires_grad_(False)

        # self.l1 = torch.nn.Linear(in_features=self.in_features, out_features=self.width * 2, bias=True)
        # self.l2 = torch.nn.Linear(in_features=self.width, out_features=self.out_features, bias=False)
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

        elif mode == 'Physics-consistent':
            # self.activation = self.cos_activaton
            self.activation = self.bessel0_activation if not self.one_d_flag else self.sin_activaton
            if green_activation_for_pcnn_2d:
                print('\n\n' + '-' * 60 + '\n' + 'Green function used as the activation!')
                self.nn_x = net_x(in_features=self.in_features, width=self.width, bias_flag=True)
                self.nn_y = net_x(in_features=self.in_features, width=self.width, bias_flag=True)
                self.alpha = torch.nn.Parameter(
                    torch.ones(size=[1])[0], requires_grad=True)
                self.beta = torch.nn.Parameter(
                    torch.ones(size=[1])[0], requires_grad=True)
                self.l2 = torch.nn.Linear(in_features=self.width, out_features=self.out_features)
                self.l2_sin = torch.nn.Linear(in_features=self.width, out_features=self.out_features)
                self.forward = self.forward_green
        # self.activation_other = torch.nn.Tanh()
        else:
            raise ValueError

    def forward(self, x):
        '''

        :param x: in shape of (num_samples, in_features)
        :return:
        '''

        y = self.activation(x @ self.w_k + self.a_k) @ self.v_k
        return y

    def forward_green(self, x):  # x in shape of ((number of samples, 2))
        x_ = self.nn_x(x[:, 0:2])  # in shape of (number of samples, width)
        y_ = self.nn_y(x[:, 0:2])  # in shape of (number of samples, width)
        real, imaginary = self.green_activation(x_, y_)
        output = self.l2(real) + self.l2_sin(imaginary)  # in shape of (number of samples, 1)
        return output

    def sin_activaton(self, x):
        # return self.activation_other(x)
        return torch.sin(x)

    def bessel0_activation(self, x):
        t = torch.ones_like(x)
        s_um = torch.ones_like(x)
        x_half_2 = (x / 2) ** 2
        for i in range(1, 40):
            t *= -x_half_2 / i ** 2
            s_um += t
        # s_um = torch.special.bessel_j0(x)
        return s_um

    def green_activation(self, x, y):
        # r = torch.linalg.norm(x, d)
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-4)  # in shape of (number of samples, width)
        h_real = torch.cos(self.alpha * r)/ r   # in shape of (number of samples, width)
        h_imaginary = torch.cos(self.beta * r)/ r
        return h_real, h_imaginary

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


class net_x(torch.nn.Module):
    def __init__(
            self, in_features=1, out_features=1, width=100, bias_flag = True
    ):
        super(net_x, self).__init__()
        self.l1 = torch.nn.Linear(in_features=in_features, out_features=width, bias=True)

    def forward(self, x):
        return self.l1(x)
