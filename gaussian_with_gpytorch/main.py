import gpytorch
import torch
import os

import tqdm

from gp_model import ExactGPModel
import numpy as np
import matplotlib.pyplot as plt


def main(training_iter=100):

    # enter into the train mode
    model.train()
    likelihood.train()

    # use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # this will include the GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * training_iter], gamma=0.1)

    # "loss" for the GPs
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=likelihood, model=model)

    loss_histoty = []
    # iter for training
    with tqdm.trange(training_iter) as pbar:
        for i in pbar:
            # zero gradients
            optimizer.zero_grad()
            # get output from the model
            output = model.forward(train_x)
            # cal the negative marginal log likelihood as loss
            loss = -mll(output, train_y)
            loss.backward()
            # if i % 10 == 0:
            #     print(' Iter %d/%d  \tLoss: %.3e \tLengthscale: %.3e \tnoise: %.3e' % (
            #         i+1, training_iter, loss.item(),
            #         model.covar_module.base_kernel.lengthscale.item(),
            #         model.likelihood.noise.item(),
            #     ))
            dic = {
                'Iter': '%d/%d' % (i+1, training_iter),
                'Loss': loss.item(),
                'Lengthscale': model.covar_module.base_kernel.lengthscale.item(),
                'noise': model.likelihood.noise.item(),
                    }
            loss_histoty.append([i, loss.item()])
            pbar.set_postfix(dic)
            optimizer.step()
            scheduler.step()

    # plot the training loss
    plot_loss(np.array(loss_histoty))

    # ---------------------------------
    # predicting on the test dataset
    # enter the eval mode
    model.eval()
    likelihood.eval()

    # test points are regularly spaced along [0, 1]
    # make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        temp_distribution = model(test_x)
        observed_pred = model.likelihood(temp_distribution)  # TODO why need to apply likelihood here?

        temp_distribution.loc - observed_pred.loc
        temp_distribution.covariance_matrix.diagonal() - observed_pred.covariance_matrix.diagonal()

        for temp in [temp_distribution, observed_pred]:
            plt.plot(test_x, temp.loc.detach().numpy(), linestyle='--')
            plt.fill_between(
                test_x,
                (temp.mean-temp.stddev*1.96).numpy(),
                (temp.mean+temp.stddev*1.96).numpy(), alpha=0.5)

        plt.plot(test_x, data_func(test_x).detach().numpy(), label='Truth')
        plt.legend(['f(x)', 'y', 'Truth'])
        plt.show()

    # ------------------------------------
    # plot the prediction
    with torch.no_grad():
        # # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

        plot_prediction(
            xtest=test_x, ytest=data_func(test_x), lower=lower, upper=upper, mu=observed_pred.mean,
            noisy=True, xtest_noise=train_x, ytest_noise=train_y,
        )

    # Auto_grad
    auto_grad(x_noise=train_x)

    # input err_propagation
    input_err_propagation(x_noise=train_x)

    # save the model
    model.state_dict()
    torch.save(model.state_dict(), 'my_gp_with_nn_model.pth')

    # restore model
    state_dict_reload = torch.load('my_gp_with_nn_model.pth')
    model_1 = ExactGPModel(train_x, train_y, likelihood)
    model_1.load_state_dict(state_dict_reload)

    return


def data_func(x):
    f = lambda a: np.sin(1. * np.pi / 1.6 * np.cos(5 + 1. * a))
    return f(x)


def plot_loss(loss_history):
    '''

    :param loss_history: [num_epoch, (epoch, loss)]
    :return:
    '''
    fig, ax = plt.subplots()
    ax.plot(loss_history[:, 0], loss_history[:, 1])
    ax.set(title="Loss", xlabel="Iterations", ylabel="Negative Log-Likelihood")
    plt.tight_layout()
    plt.show()


def plot_prediction(xtest, ytest, mu, lower, upper, noisy=None, xtest_noise=None, ytest_noise=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if noisy:
        ax.scatter(xtest_noise, ytest_noise, marker="o", s=30, color="tab:orange", label="Noisy Test Data")
    else:
        ax.scatter(xtest, ytest, marker="o", s=30, color="tab:orange", label="Noisy Test Data")
    ax.plot(xtest, ytest, color="black", linestyle="-", label="True Function")
    ax.plot(
        xtest,
        mu.ravel(),
        color="Blue",
        linestyle="--",
        linewidth=3,
        label="Predictive Mean",
    )
    ax.fill_between(
        xtest.ravel(),
        lower,
        upper,
        alpha=0.4,
        color="tab:blue",
        label=f" 95% Confidence Interval",
    )
    ax.plot(xtest, lower, linestyle="--", color="tab:blue")
    ax.plot(xtest, upper, linestyle="--", color="tab:blue")
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.show()
    return fig, ax


def mean_f(x):
    return likelihood(model(x)).mean.sum()


def var_f(x):
    return likelihood(model(x)).var.sum()


def mean_df(x):
    return torch.autograd.functional.jacobian(mean_f, x, create_graph=True).sum()


def var_df(x):
    return torch.autograd.functional.jacobian(var_f, x, create_graph=True).sum()


def auto_grad(x_noise):
    x = torch.autograd.Variable(torch.tensor(x_noise), requires_grad=True)
    dy_dx_f = torch.autograd.functional.jacobian(mean_f, x)
    dy_dx2_f = torch.autograd.functional.jacobian(mean_df, x)
    dy_dx2_f_hessian = torch.autograd.functional.hessian(mean_f, x).diagonal()

    mu = likelihood(model(x)).mean

    fig, ax = plt.subplots(figsize=(6, 3))
    plt.plot(x.detach().numpy(), mu.detach().numpy())
    plt.plot(x.detach().numpy(), dy_dx_f.detach().numpy())
    plt.plot(x.detach().numpy(), dy_dx2_f.detach().numpy())
    plt.plot(x.detach().numpy(), dy_dx2_f_hessian.detach().numpy(), linestyle='--')
    plt.legend([r'$\mu$', '1st', '2nd', '2nd_hessian'])
    plt.tight_layout()
    plt.show()
    return dy_dx_f, dy_dx2_f


def input_err_propagation(x_noise):
    dy_dx_f, dy_dx2_f = auto_grad(x_noise)
    dy_dx_f = dy_dx_f.reshape(1, -1)
    x = torch.autograd.Variable(torch.tensor(x_noise), requires_grad=True)
    # input_cov = (x_noise ** 2).reshape(1, -1)
    x = x.reshape(-1, 1)
    input_cov = x @ x.t()

    f_pred = likelihood(model(x))
    mu, std = f_pred.mean, f_pred.stddev

    var_corr = dy_dx_f.matmul(input_cov).matmul(dy_dx_f.t()).diagonal()
    std_corr = var_corr.sqrt()
    egp_std = std + std_corr
    egp_lower = mu - 1.96 * egp_std
    egp_upper = mu + 1.96 * egp_std

    plot_prediction(
        xtest=x_noise, ytest=data_func(x_noise), lower=egp_lower.detach().numpy(),
        upper=egp_upper.detach().numpy(), mu=mu.detach().numpy())
    return


if __name__ == '__main__':
    # data generation
    train_x = torch.linspace(0, 1, 200)
    train_y = data_func(train_x) + torch.randn(train_x.size()) * np.sqrt(0.04)
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    main()