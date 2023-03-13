import matplotlib.pyplot as plt
import numpy as np
import torch
from NN_models.network import net_basic
from NN_models.train_nn import plot_loss
from dataset_gen_gaussian import gaussian_data_gen, data_load

'''
    Here is a script referred to  
        [1] S. Ranftl, “A connection between probability, physics and neural networks,” 2022, [Online]. 
            Available: http://arxiv.org/abs/2209.12737.
            
        [2] C. Albert, “Physics-informed transfer path analysis with parameter estimation using Gaussian processes,” 
            Proc. Int. Congr. Acoust., vol. 2019-September, no. 1, pp. 459–466, 2019, doi: 10.18154/RWTH-CONV-238988.
'''


def main(ratio=0.3):
    x, y, y_noise, index = data_load(save_path='xy_data')
    mode_list = [ 'vanilla', 'physics_informed', 'physics_constrained', ]
    loss_dic = {}
    num_saple = len(x)
    index = index[:int(num_saple*ratio)]
    # index = np.random.permutation(np.arange(0, num_saple))[:int(num_saple*ratio)]
    # index = np.arange(0, num_saple)[:int(num_saple*ratio)]
    for mode in mode_list:
        print('\n\n' + '=' * 60 + '\n' + '\tMode: %s' % mode + '\n')
        loss_dic[mode] = single_train(x=x[index], y=y_noise[index], mode=mode, width=30)
    #
    plot_loss(mode_list=mode_list, loss_dic=loss_dic)
    #
    test( mode_list=mode_list)


def single_train(x, y, num_epoch=20000, mode='physics_constrained', width=10):
    # x, y = dataset_gen()
    model = net_basic(mode=mode, width=width)
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
            y_pre, ddy = model.forward_ddy(x_tensor)
            loss = loss_operator(y_pre, y_tensor) + \
                   1. * loss_operator(ddy, -model.nu**2. * y_tensor)
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
            loss = loss_operator(y_pre, y_tensor)
            line = 'Epoch %d The current loss is %.3e %s' % (epoch, loss.item(), mode)
            if 'informed' in mode:
                line += ' nu=%.2f' % model.nu.item()
            print(line)
            loss_list.append([epoch, loss.item()])
    torch.save(model, f='%s.pt' % mode)
    return np.array(loss_list)


def test(mode_list: list, ratio: float = 0.2):
    '''
            Check the model's performance on the test datasets
    :param x:
    :param y:
    :param mode_list:
    :param num_points:
    :return:
    '''

    x, y, y_noise, index = data_load(save_path='xy_data')
    index = index[: int(len(x)*ratio)]
    data_dic = {}
    data_dic['TruthWithNoise'] = [x[index], y_noise[index]]
    for mode in mode_list:
        model = torch.load('%s.pt' % mode)
        x_tensor = torch.from_numpy(x).float()
        y_pre = model.forward(x_tensor).detach().numpy()
        data_dic[mode] = [x, y_noise, y_pre]
    xx = np.linspace(0, 1.0, 100)
    plt.plot(xx, y, 'k.', label='TruthWithoutNoise')
    plot_comparison(data_dic)


def plot_comparison(data_dic):
    keys = list(data_dic.keys())
    for key in keys:
        if key == 'TruthWithNoise':
            x, y = data_dic[key][0], data_dic[key][1]
            num_point = len(x)
            index = list(range(num_point))
            plt.scatter(x[index].flatten(), y[index].flatten(), color='r', label='TruthWithNoises')
            continue
        x, y_pre = data_dic[key][0], data_dic[key][2]
        plt.plot(x, y_pre, label=key)
    plt.legend()
    plt.title('Comparison')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
