import torch
from network import net_basic
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset_gen_gaussian import gaussian_data_gen


def main():
    x, y = dataset_gen(num_points=10, noise=0.2)
    loss_list = []
    mode_list = ['physics_informed', 'vallina',  'physics_constrained']
    for mode in mode_list:
        print('\n\n' + '='*60 + '\n' + '\tMode: %s' % mode + '\n')
        loss_list.append(single_train(x=x, y=y, mode=mode))
    plot_loss(mode_list=mode_list, loss_array=np.array(loss_list))
    x, y = dataset_gen(num_points=10, noise=0.2)
    test(x=x, y=y, mode_list=mode_list)


def single_train(x, y, num_epoch=10000, mode='physics_constrained'):
    # x, y = dataset_gen()
    model = net_basic(mode=mode, width=10)
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
            loss = loss_operator(y_pre, y_tensor) + 1.0 * loss_operator(physics_item, torch.zeros_like(physics_item))
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
            print('Epoch %d The current loss is %.3e %s' % (epoch, loss.item(), mode))
            loss_list.append([epoch, loss.item()])
    torch.save(model, f='%s.pt' % mode)
    return loss_list


def dataset_gen(num_points=10, noise=0.2):
    x = np.linspace(0, np.pi*2., num_points)
    y = np.sin(x + 0.5) + np.random.random(num_points)*noise
    # x, y_list = gaussian_data_gen(num_samples=1, num_points=num_points)
    # y = y_list[0] + np.random.random(num_points)*noise
    return np.array(x[:, np.newaxis]), np.array(y[:, np.newaxis])


def test(x, y, mode_list: list, ):
    # x, y = dataset_gen()
    data_dic = {}
    for mode in mode_list:
        model = torch.load('%s.pt' % mode)
        x_tensor, y_tensor = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        y_pre = model.forward(x_tensor).detach().numpy()
        data_dic[mode] = [x, y, y_pre]
    plot_comparison(data_dic)


def plot_comparison(data_dic):
    keys = list(data_dic.keys())
    x, y = data_dic[keys[0]][0], data_dic[keys[0]][1]
    num_point = len(x)
    index = list(range(0, num_point, 20)) if num_point>20 else list(range(num_point))
    plt.scatter(x[index].flatten(), y[index].flatten(), color='r', label='GroudTruth')
    for key in keys:
        x, y_pre = data_dic[key][0], data_dic[key][2]
        plt.plot(x, y_pre, label=key)
    plt.legend()
    plt.title('Comparison')
    plt.tight_layout()
    plt.show()


def plot_loss(mode_list, loss_array):
    n = len(mode_list)
    for i in range(n):
        plt.plot(loss_array[i][:, 0], loss_array[i][:, 1], label=mode_list[i])
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()