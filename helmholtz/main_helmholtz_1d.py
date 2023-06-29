import numpy as np
import torch
from NN_model import net_basic
from train_nn import single_train, plot_loss
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


np.random.seed(10000)
fig_save_path = 'xy_data'
mode_list = ['Vanilla', 'Physics-informed', 'Physics-consistent', ]
# mode_list = [ 'Physics-informed' ]


# plot configuration
mpl.rcParams['figure.dpi'] = 100
# fix random seeds
axes = {'labelsize': 'large'}
font = {'family': 'serif',
        'weight': 'normal',
        'size': 17}
legend = {'fontsize': 15}
lines = {'linewidth': 3,
         'markersize': 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)


# datasets preparation
A = 1.0
omega = 6.
phi = np.pi/3.
nx = 101
x = np.linspace(0., 1.0, nx)
noises_amptitude = 0.04
y = A * np.sin(omega*x + phi)
y_noise = y + np.random.randn(nx) *noises_amptitude
index_random = np.random.permutation(list(range(nx)))
index_train = index_random[:int(0.3 * nx)]
index_test = index_random[-int(0.2 * nx):]
plt.plot(x, y, label='Truth')
plt.scatter(x[index_train], y_noise[index_train], c='g', marker='x', label='Train sets')
plt.scatter(x[index_test], y_noise[index_test], c='k', marker='o', label='Validation sets')
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
fig = plt.gcf()
fig.savefig('%s/1d_datasets.png' % fig_save_path, dpi=200)
plt.show()
plt.close()


def main(
        num_epoch=20000,  # 20000,
        save_path='xy_data'):
    loss_dic = {}
    x_temp = x[:, np.newaxis]
    y_noise_temp = y_noise[:, np.newaxis]
    nu_dic = {}
    for mode in mode_list:
        print('\n\n' + '=' * 60 + '\n' + '\tMode: %s' % mode + '\n')
        loss_dic[mode], nu_dic[mode] = single_train(
            x=x_temp[index_train], y=y_noise_temp[index_train],
            x_validation=x_temp[index_test], y_validation=y_noise_temp[index_test],
            mode=mode, width=100, num_epoch=num_epoch, one_d_flag=True)
    test_trained_model()
    #
    plot_loss(mode_list=mode_list, loss_dic=loss_dic, save_path=save_path, ond_d_flag=True)
    #
    # plot the evolution of nu
    plot_nu(loss_dic['Physics-informed'], nu_dic['Physics-informed'], ond_d_flag=True)


def test_trained_model():
    prediction = []

    plot_index = range(0, nx, nx//15)
    plt.scatter(x[plot_index], y[plot_index], c='k', label='Truth', zorder=10)
    for mode in mode_list:
        model = torch.load('%s_1d.pt' % mode)

        with torch.no_grad():
            prediction.append(model.forward(torch.from_numpy(x[:, np.newaxis]).float()).numpy().reshape(-1))
        plt.plot(x, prediction[-1], label=mode)

    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig('%s/1d_prediction.png' % fig_save_path, dpi=200)
    plt.show()
    plt.close()

    return


def plot_nu(loss, nu_arr, ond_d_flag=False):
    epoch = loss[:, 0]
    plt.plot(epoch/1e3, nu_arr)
    plt.xlabel('Epoch/1e3')
    plt.ylabel('$k$')
    plt.scatter(epoch[-1]/1e3, nu_arr[-1], c ='r', zorder=10, s=100 ,edgecolors='k')
    plt.plot(epoch/1e3, nu_arr[-1]*np.ones(len(epoch)), '--', c ='k', zorder=10)
    plt.xlim([0, epoch[-1]/1e3 + 0.5])
    # plt.text(epoch[-1]/1e3, nu_arr[-1], '$k=%.2f$' % nu_arr[-1], bbox=dict(facecolor='red', alpha=0.5))
    plt.annotate(
        '$k=%.2f$' % nu_arr[-1], (epoch[-1]/1e3-2.0, nu_arr[-1]-0.6),
        bbox=dict(facecolor='gray', alpha=0.5), fontsize=20)
    plt.grid()
    plt.tight_layout()
    fig = plt.gcf()
    name = os.path.join(fig_save_path, 'nu_evolution.png' if not ond_d_flag else 'nu_evolution_1d.png')
    fig.savefig(name, dpi=200)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    print()