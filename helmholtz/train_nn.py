import os.path
from NN_model import net_basic
import torch
import numpy as np
from utils_general.utils import echo
import matplotlib.pyplot as plt


def single_train(
        x, y,
        x_test, y_test,
        num_epoch,
        mode='Physics-constrained', width=10,
        patience=50, # used to be 20
        one_d_flag=False,
        green_activation_for_pcnn_2d=False,
        x_physics = None, y_physics = None, generalization_test_flag = False
):
    # x, y = dataset_gen()
    model = net_basic(
        in_features=len(x[0]), out_features=len(y[0]), mode=mode, width=width, one_d_flag=one_d_flag,
        green_activation_for_pcnn_2d=green_activation_for_pcnn_2d)
    optim = torch.optim.Adam(model.parameters())
    loss_operator = torch.nn.MSELoss()
    x_tensor, y_tensor = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    try_num = 0
    max_err = 1e5
    loss_list = []
    nu_list = []
    name_model = mode
    if one_d_flag:
        name_model += '_1d'
    if generalization_test_flag:
        name_model += '_generalization'
    name_model += '.pt'
    model_state = model.state_dict()
    for epoch in range(num_epoch):
        optim.zero_grad()
        if mode == 'Vanilla' or mode == 'Physics-constrained':
            y_pre = model.forward(x_tensor)
            loss = loss_operator(y_tensor, y_pre)
        elif 'informed' in mode:
            if not generalization_test_flag:
                y_pre, ddy_pre = model.forward_ddy(x_tensor)
                loss = loss_operator(y_pre, y_tensor) + \
                       0.00001 * loss_operator(ddy_pre.sum(-1, keepdim=True), -model.nu**2. * y_pre)
                   # 0.0001 * loss_operator(ddy_pre.sum(-1, keepdim=True), -model.nu**2. * y_pre)
            else: # the data term and the physics term use different data sets
                x_physics_tensor, y_physics_tensor = torch.from_numpy(x_physics).float(), \
                                                     torch.from_numpy(y_physics).float()

                y_pre, ddy_pre = model.forward_ddy(x_tensor)
                y_pre_physics, ddy_pre_physics = model.forward_ddy(x_physics_tensor)

                loss = loss_operator(y_pre, y_tensor) + \
                       0.00001 * loss_operator(ddy_pre.sum(-1, keepdim=True), -model.nu**2. * y_pre) + \
                       0.00001 * loss_operator(ddy_pre_physics.sum(-1, keepdim=True), -model.nu**2. * y_pre_physics)

            '''
                NOTE: the lambda here should be carefully selected according to 
                 [1] why PINNS fail to train: https://www.sciencedirect.com/science/article/pii/S002199912100663X
            '''
        else:
            raise ValueError('No mode %s, \n please check the mode!' % mode)

        loss.backward()
        optim.step()
        if epoch % 100 == 0:
            loss = loss_operator(y_pre, y_tensor)
            y_pre_test = model.forward(x_test_tensor)
            loss_test = loss_operator(y_pre_test, y_test_tensor)
            line = 'Epoch %d The current loss is %.3e test_loss: %.3e %s' % (epoch, loss.item(), loss_test.item(), mode)
            if max_err>loss_test:
                max_err = loss_test
                try_num = 0
                line += ' improved!'
                # model_state = model.state_dict()
                torch.save(model, f=name_model)
            else:
                try_num += 1
                line += ' NoImpr.'
            if 'Physics' in mode:
                line += ' nu=%.2f' % model.nu.item()
                nu_list.append(model.nu.item())
            print(line)
            loss_list.append([epoch, loss.item(), loss_test.item()])
            if try_num >= patience:
                break

    echo('model saved as %s' % name_model)
    return np.array(loss_list), np.array(nu_list)


def plot_loss(mode_list, loss_dic: dict, save_path: str, ond_d_flag=False,
              generalization_test_flag=False):
    color_list = ['tab:blue', 'tab:orange', 'tab:green']
    for i, mode in enumerate(mode_list):
        epoch = loss_dic[mode][:, 0]
        loss = loss_dic[mode][:, 1]
        loss_true = loss_dic[mode][:, 2]
        plt.plot(epoch/1e3, loss, c=color_list[i], label=mode)
        plt.plot(epoch/1e3, loss_true, '--', c=color_list[i], zorder=10)
    plt.yscale('log')
    plt.xlabel('Epoch/1e3')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()

    name_fig = 'training_loss'
    if ond_d_flag:
        name_fig += '_1d'
    if generalization_test_flag:
        name_fig += '_generalization'
    name_fig += '.png'

    name = os.path.join(save_path, name_fig)
    fig.savefig(name, dpi=200)
    plt.show()
    plt.close()