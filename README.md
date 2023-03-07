# gaussian_nn_tugraz
This is a repository of some researches about the link between Gaussian Process and the NN.

We referred to the [work](https://arxiv.org/abs/2209.12737).

- We compared the performance of Vanilla, PINN and physics-constrained network in this work (PCNN).
- There is no doubt that the PCNN performs best as the activation function is constrained like the physics kernel 
in the Gaussian Process mentioned in [this work](https://arxiv.org/pdf/1905.07907.pdf).
- The noise is in random normal distribution with a scale of 0.15.

- From the results, we can see that, with 
    - with a proper activation function or physics (informed or constrained), like the PCNN, the model perform
better against the noises.
    - with a constrained proper activation function, the PCNN performs the best.

![The training loss](./figs/loss_train.png)
![The training loss](./figs/prediction.png)