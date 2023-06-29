import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_nn import single_train

np.random.seed(10000)
torch.manual_seed(10001)
fig_save_path = 'xy_data'
mode_list = ['Vanilla', 'Physics-informed', 'Physics-consistent', ]
# mode_list = [ 'Physics-consistent' ]


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
phi = np.pi / 3.
nx = 101
x = np.linspace(0., 1.0, nx)
noises_amptitude = 0.04
y = A * np.sin(omega * x + phi)
y_noise = y + np.random.randn(nx) * noises_amptitude
index_equidistant = np.arange(0, nx, nx // 20)
number_points = len(index_equidistant)
index_train_1 = index_equidistant[:int(0.5 * number_points)]  # data + physics
index_train_2 = index_equidistant[int(0.5 * number_points):int(0.7 * number_points)]  # physics
index_validation = np.random.permutation(list(range(nx))[:int(0.7 * number_points)])[
                   :int(0.3 * nx)]  # this is used in the validation
index_test = index_equidistant[-int(0.3 * number_points):]  # this is used in the test after training

plt.plot(x, y, label='Truth')
plt.scatter(x[index_train_1], y_noise[index_train_1], c='g', marker='x', label='Train sets')
plt.scatter(x[index_train_2], y_noise[index_train_2], c='g', marker='x')
plt.scatter(x[index_test], y_noise[index_test], c='k', marker='o', label='Test sets')

plt.fill_betweenx(
    y=[-1.5, 1.5], x1=[-0.1, -0.1], x2=[x[index_train_1[-1]], x[index_train_1[-1]]],
    color='pink', alpha=0.5, )
plt.fill_betweenx(
    y=[-1.5, 1.5], x1=[x[index_train_1[-1]], x[index_train_1[-1]]],
    x2=[x[index_train_2[-1]], x[index_train_2[-1]]],
    color='gray', alpha=0.5, )

plt.text(x=0.0, y=0., s=r'$\mathcal{L}_d + \mathcal{L}_p$', fontsize=20)
plt.text(x=0.48, y=0., s=r'$\mathcal{L}_p$', fontsize=20)
plt.xlim([-0.05, 1.05])
plt.ylim([-1.2, 1.2])
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
fig = plt.gcf()
fig.savefig('%s/1d_datasets_physics_data.png' % fig_save_path, dpi=200)
plt.show()
plt.close()


def main(
        num_epoch=20000,  # 20000,
        save_path='xy_data',
        patience = 50):
    loss_dic = {}
    x_temp = x[:, np.newaxis]
    y_noise_temp = y_noise[:, np.newaxis]
    nu_dic = {}
    for mode in mode_list:
        print('\n\n' + '=' * 60 + '\n' + '\tMode: %s' % mode + '\n')
        loss_dic[mode], nu_dic[mode] = single_train(
            x=x_temp[index_train_1], y=y_noise_temp[index_train_1],
            x_physics=x_temp[index_train_2] if 'informed' in mode else None,
            y_physics=y_noise_temp[index_train_2] if 'informed' in mode else None,
            generalization_test_flag=True,
            x_validation=x_temp[index_validation], y_validation=y_noise_temp[index_validation],
            x_test=x_temp[index_test], y_test=y_noise_temp[index_test],
            mode=mode, width=100, num_epoch=num_epoch, one_d_flag=True, patience=patience)
    # plot the model prediction
    test_trained_model(
        generalization_test_flag=True)
    # plot the training process
    plot_loss(mode_list=mode_list, loss_dic=loss_dic, save_path=save_path,
              ond_d_flag=True,

              generalization_test_flag=True)
    #
    # plot the evolution of nu
    plot_nu(loss_dic['Physics-informed'], nu_dic['Physics-informed'],
            ond_d_flag=True,
            generalization_test_flag=True)


def test_trained_model(generalization_test_flag=False):
    prediction = []

    plot_index = range(0, nx, nx // 15)
    plt.scatter(x[plot_index], y[plot_index], c='k', label='Truth', zorder=10)
    for mode in mode_list:
        model = torch.load('%s_1d_generalization.pt' % mode)

        with torch.no_grad():
            prediction.append(model.forward(torch.from_numpy(x[:, np.newaxis]).float()).numpy().reshape(-1))
        plt.plot(x, prediction[-1], label=mode)

    plt.fill_betweenx(
        y=[-1.5, 1.5], x1=[-0.1, -0.1], x2=[x[index_train_1[-1]], x[index_train_1[-1]]],
        color='pink', alpha=0.5, )
    plt.fill_betweenx(
        y=[-1.5, 1.5], x1=[x[index_train_1[-1]], x[index_train_1[-1]]],
        x2=[x[index_train_2[-1]], x[index_train_2[-1]]],
        color='gray', alpha=0.5, )
    plt.text(x=0.0, y=0., s=r'$\mathcal{L}_d + \mathcal{L}_p$', fontsize=20)
    plt.text(x=0.48, y=0., s=r'$\mathcal{L}_p$', fontsize=20)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-1.2, 1.2])

    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.tight_layout()
    fig = plt.gcf()

    name_fig = '1d_prediction'
    if generalization_test_flag:
        name_fig += '_generalization'
    name_fig += '.png'

    fig.savefig('%s/%s.png' % (fig_save_path, name_fig), dpi=200)
    plt.show()
    plt.close()

    return


def plot_nu(loss, nu_arr, ond_d_flag=False, generalization_test_flag=False):
    epoch = loss[:, 0]
    plt.plot(epoch / 1e3, nu_arr)
    plt.xlabel('Epoch/1e3')
    plt.ylabel('$k$')
    plt.scatter(epoch[-1] / 1e3, nu_arr[-1], c='r', zorder=10, s=100, edgecolors='k')
    plt.plot(epoch / 1e3, nu_arr[-1] * np.ones(len(epoch)), '--', c='k', zorder=10)
    plt.xlim([0, epoch[-1] / 1e3 + 0.5])
    # plt.text(epoch[-1]/1e3, nu_arr[-1], '$k=%.2f$' % nu_arr[-1], bbox=dict(facecolor='red', alpha=0.5))
    plt.annotate(
        '$k=%.2f$' % nu_arr[-1], (epoch[-1] / 1e3 - 2.0, nu_arr[-1] - 0.6),
        bbox=dict(facecolor='gray', alpha=0.5), fontsize=20)
    plt.grid()
    plt.tight_layout()
    fig = plt.gcf()
    name_fig = 'nu_evolution'
    if ond_d_flag:
        name_fig += '_1d'
    if generalization_test_flag:
        name_fig += '_generalization'
    name_fig += '.png'
    name = os.path.join(fig_save_path, name_fig)
    fig.savefig(name, dpi=200)
    plt.show()
    plt.close()


def plot_loss(mode_list, loss_dic: dict, save_path: str, ond_d_flag=False,
              generalization_test_flag=False):
    color_list = ['tab:blue', 'tab:orange', 'tab:green']
    for i, mode in enumerate(mode_list):
        epoch = loss_dic[mode][:, 0]
        loss = loss_dic[mode][:, 1]
        loss_validation = loss_dic[mode][:, 2]
        loss_test = loss_dic[mode][:, 3]
        plt.plot(epoch / 1e3, loss, linestyle='solid', c=color_list[i], label=mode)
        plt.plot(epoch / 1e3, loss_validation, linestyle='--', c=color_list[i], zorder=10)
        plt.plot(epoch / 1e3, loss_test, linestyle='dotted', c=color_list[i], zorder=11)

        # plot the line of convergence
        if 'Physics' in mode:
            min_vali_loss_index = np.armin(loss_validation)

            epoch_max = max(epoch)/1e3
            plt.plot([epoch_max, epoch_max], [1e-4, 1e-1],
                     c='k', linestyle='dashdot', zorder=12)
            plt.scatter(
                x= [epoch_max, epoch_max], y=[loss_validation[min_vali_loss_index], loss[min_vali_loss_index]], color = 'r')

    plt.yscale('log')
    plt.xlabel('Epoch/1e3')
    plt.ylabel('Error')
    plt.legend()
    plt.ylim([3e-4, 2e1])
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


if __name__ == '__main__':
    main()
