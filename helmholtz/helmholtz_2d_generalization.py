import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from helmholtz_2d_data_gen import data_load
from kernel_2d import input_2d
from train_nn import single_train, plot_loss
from utils_general.utils import echo

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


def main(
        num_epoch=20000,  # 20000,
        ratio=0.2,
        save_path='xy_data'):
    x, y, y_noise, index = data_load(save_path=save_path)
    y = y[:, np.newaxis]
    num_samples = len(x)
    index_train = index[: int(num_samples * ratio)]
    index_test = index[-int(num_samples * ratio):]
    y_noise = y_noise[:, np.newaxis]
    # y_noise = y[:, np.newaxis]
    # mode_list = ['Vanilla', 'Physics-informed', 'Physics-constrained', ]
    mode_list = ['Physics-constrained', ]
    green_activation_for_pcnn_2d = True
    loss_dic = {}
    nu_dic = {}
    for mode in mode_list:
        print('\n\n' + '=' * 60 + '\n' + '\tMode: %s' % mode + '\n')
        loss_dic[mode], nu_dic[mode] = single_train(
            x=x[index_train], y=y_noise[index_train],
            x_test=x[index_test], y_test=y_noise[index_test],
            mode=mode, width=100, num_epoch=num_epoch,
            green_activation_for_pcnn_2d=green_activation_for_pcnn_2d)
        test_single_trained_model(mode=mode)
    #
    plot_loss(mode_list=mode_list, loss_dic=loss_dic, save_path=save_path)
    #
    # test(x=x, y=y, mode_list=mode_list)


def test_single_trained_model(n=20, mode='Vanilla', fig_save_path='xy_data'):
    x_train, y_train, y_noise, index = data_load(save_path='xy_data')

    model = torch.load('%s.pt' % mode)

    x_test, X, Y = input_2d(min_=-1, max_=1, num=n)
    with torch.no_grad():
        z = model.forward(torch.from_numpy(x_test).float()).numpy()
        plt.contourf(X, Y, z.reshape(n, n), cmap = 'RdBu',
                             vmin = -3.0,
                             vmax = 2.4)
    plt.colorbar()
    plt.tight_layout()
    fig_name = os.path.join(fig_save_path, '%s.png' % mode)
    if 'informed' in mode:
        plt.title(r'$\nu=%.2f$' % model.nu)
    plt.savefig(fig_name, dpi=200)
    echo('fig saved as %s' % fig_name)
    plt.close()

    # error plot
    plt.contourf(X, Y, np.abs(z.reshape(n, n) - y_train.reshape(n, n)), cmap = 'RdBu',
                             vmin = -3.0,
                             vmax = 2.4)
    plt.colorbar()
    # plt.title('error_%s' % mode)
    plt.tight_layout()
    fig_name = os.path.join(fig_save_path, '%s_error.png' % mode)
    plt.savefig(fig_name, dpi=200)
    echo('fig saved as %s' % fig_name)
    plt.close()

    array_save_name = os.path.join(fig_save_path, '%s_pre.npy' % mode)
    np.save(array_save_name, z)
    echo('The prediction of %s is saved in %s' % (mode, array_save_name))

    return


def plot_cut_line(n=20, save_path='xy_data'):
    mode_list = ['Vanilla', 'Physics-informed', 'Physics-constrained']
    x_train, y_train, y_noise, index = data_load(save_path='xy_data')
    x_test, X, Y = input_2d(min_=-1, max_=1, num=n)
    z_list = [np.load(os.path.join(save_path, '%s_pre.npy' % mode)) for mode in mode_list]
    # plot the cut line
    for cut_n in [0, 10, 19]:
        posi_y = (cut_n / (n - 1) * 2. - 1.)
        y_train_cut = y_train.reshape(n, n)[cut_n, :]
        plt.plot(X[0], y_train_cut, 'k.', label='Truth')
        # y_noise_cut = y_noise.reshape(n, n)[cut_n, :]
        # plt.plot(X[0], y_noise_cut, label='Training_noise')
        for i, mode in enumerate(mode_list):
            z_cut = z_list[i].reshape(n, n)[cut_n, :]
            plt.plot(X[0], z_cut, label='%s' % mode)
        plt.title('Cut at the y=%.2f' % posi_y)
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Predicted value')
        plt.tight_layout()
        fig_name = os.path.join(save_path, 'error_cut_%.2f.png' % (posi_y))
        plt.savefig(fig_name, dpi=200)
        echo('fig saved as %s' % fig_name)
        plt.close()


if __name__ == '__main__':
    main()
    plot_cut_line()
