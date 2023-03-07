import matplotlib.pyplot as plt
import numpy as np
import torch
from network import net_basic

'''
    Here is a script referred to  
        [1] S. Ranftl, “A connection between probability, physics and neural networks,” 2022, [Online]. 
            Available: http://arxiv.org/abs/2209.12737.
            
        [2] C. Albert, “Physics-informed transfer path analysis with parameter estimation using Gaussian processes,” 
            Proc. Int. Congr. Acoust., vol. 2019-September, no. 1, pp. 459–466, 2019, doi: 10.18154/RWTH-CONV-238988.
'''


def main():
    x, y = dataset_gen(num_points=10, noise=0.15)
    mode_list = ['vallina', 'physics_informed', 'physics_constrained']
    loss_dic = {}
    for mode in mode_list:
        print('\n\n' + '=' * 60 + '\n' + '\tMode: %s' % mode + '\n')
        loss_dic[mode] = single_train(x=x, y=y, mode=mode, width=10)
    #
    plot_loss(mode_list=mode_list, loss_dic=loss_dic)
    #
    test(x=x, y=y, mode_list=mode_list)


def single_train(x, y, num_epoch=50000, mode='physics_constrained', width=10):
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
            y_pre, physics_item = model.forward_ddy(x_tensor)
            loss = loss_operator(y_pre, y_tensor) + \
                   1.0 * loss_operator(physics_item, torch.zeros_like(physics_item))
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
            loss = loss_operator(y_pre, y_tensor)
            print('Epoch %d The current loss is %.3e %s' % (epoch, loss.item(), mode))
            loss_list.append([epoch, loss.item()])
    torch.save(model, f='%s.pt' % mode)
    return np.array(loss_list)


def dataset_gen(num_points=10, noise=0.05):
    x = np.linspace(0, np.pi * 2., num_points)
    y = data_function(x) + np.random.normal(size=num_points) * noise
    # x, y_list = gaussian_data_gen(num_samples=1, num_points=num_points)
    # y = y_list[0] + np.random.random(num_points)*noise
    return np.array(x[:, np.newaxis]), np.array(y[:, np.newaxis])


def data_function(x):
    return np.sin(x + 0.5)


def test(x, y, mode_list: list, num_points=50):
    # x, y = dataset_gen()
    xx = np.linspace(0, np.pi * 2., num_points).reshape(-1, 1)
    data_dic = {}
    data_dic['TruthWithNoise'] = [x, y]
    for mode in mode_list:
        model = torch.load('%s.pt' % mode)
        x_tensor = torch.from_numpy(xx).float()
        y_pre = model.forward(x_tensor).detach().numpy()
        data_dic[mode] = [xx, data_function(xx), y_pre]
    xx = np.linspace(0, np.pi * 2., 100)
    plt.plot(xx, np.sin(xx + 0.5), 'k.', label='TruthWithoutNoise')
    plot_comparison(data_dic)


def plot_comparison(data_dic):
    keys = list(data_dic.keys())
    for key in keys:
        if key == 'TruthWithNoise':
            x, y = data_dic[key][0], data_dic[key][1]
            num_point = len(x)
            index = list(range(0, num_point, 20)) if num_point > 20 else list(range(num_point))
            plt.scatter(x[index].flatten(), y[index].flatten(), color='r', label='TruthWithNoises')
            continue
        x, y_pre = data_dic[key][0], data_dic[key][2]
        plt.plot(x, y_pre, label=key)
    plt.legend()
    plt.title('Comparison')
    plt.tight_layout()
    plt.show()


def plot_loss(mode_list, loss_dic: dict):
    for mode in mode_list:
        epoch = loss_dic[mode][:, 0]
        loss = loss_dic[mode][:, 1]
        plt.plot(epoch, loss, label=mode)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
