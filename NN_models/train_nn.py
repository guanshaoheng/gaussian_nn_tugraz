from NN_models.network import net_basic
import torch
import numpy as np
from utils_general.utils import echo
import matplotlib.pyplot as plt


def single_train(x, y, num_epoch, mode='physics_constrained', width=10):
    # x, y = dataset_gen()
    model = net_basic(in_features=len(x[0]), out_features=len(y[0]), mode=mode, width=width)
    optim = torch.optim.Adam(model.parameters())
    loss_operator = torch.nn.MSELoss()
    x_tensor, y_tensor = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    loss_list = []
    for epoch in range(num_epoch):
        optim.zero_grad()
        if mode != 'physics_informed':
            y_pre = model.forward(x_tensor)
            loss = loss_operator(y_tensor, y_pre)
        else:
            y_pre, ddy_pre = model.forward_ddy(x_tensor)
            loss = loss_operator(y_pre, y_tensor) + \
                   1. * loss_operator(ddy_pre.sum(-1, keepdim=True), -model.nu**2. * y_tensor)
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
            loss = loss_operator(y_pre, y_tensor)
            line = 'Epoch %d The current loss is %.3e %s' % (epoch, loss.item(), mode)
            if mode == 'physics_informed':
                line += ' nu=%.2f' % model.nu.item()
            print(line)
            loss_list.append([epoch, loss.item()])
    name_model = '%s.pt' % mode
    torch.save(model, f=name_model)
    echo('model saved as %s' % name_model)
    return np.array(loss_list)


def plot_loss(mode_list, loss_dic: dict):
    for mode in mode_list:
        epoch = loss_dic[mode][:, 0]
        loss = loss_dic[mode][:, 1]
        plt.plot(epoch, loss, label=mode)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()