# gaussian_nn_tugraz
This is a repository of some researches about the link between Gaussian Process and the NN.

We referred to the [work](https://arxiv.org/abs/2209.12737).

- We compared the performance of Vanilla, PINN and physics-constrained network in this work (PCNN).
- There is no doubt that the PCNN performs best as the activation function is constrained like the physics kernel 
in the Gaussian Process mentioned in [this work](https://arxiv.org/pdf/1905.07907.pdf).

# 1-D Helmholtz equations
- The noise is in random normal distribution with a scale of 0.10.

- From the results, we can see that, with 
    - with a proper activation function or physics (informed or constrained), like the PCNN, the model perform
better against the noises.
    - in **vanilla** network, the activation is <span style="color:red">**ReLU()**</span>, 
  in **PINN** the activation is <span style="color:red">**Sigmoid()**</span>, while in the
  **PCNN** work the activation is <span style="color:red">**sin()**</span>
    - with a constrained proper activation function, the PCNN performs the best.
    - compared with the results of 'vanilla' network, the PINN and PCNN behave better against the noise.
    
| ![space-1.jpg](./figs/loss_train.png) | ![space-1.jpg](./figs/prediction.png) |
|:--:| :--:| 
| **training loss (1D)** |**prediction (1D)**|


# 2-D Helmholtz equations
First, the training datasets is generated via Random Gaussian Process, whose kernel is 
the [Bessel function](https://en.wikipedia.org/wiki/Bessel_function)
![equation](https://latex.codecogs.com/svg.image?J_0(k\|\mathbf{x}&space;-\mathbf{x}'&space;\|)), 
and **noise=0.1**.

The data-driven term of the loss function is 
![equation](https://latex.codecogs.com/svg.image?\|y-\hat{y}\|^2) which is the loss function 
for **vanilla NN** and **physics-constrained NN**. 
The physics term of the loss function is 
![equation](https://latex.codecogs.com/svg.image?\Delta&space;f&plus;\nu^2f). Then the loss function 
in the **PINN** work is 
![eqiation](https://latex.codecogs.com/svg.image?\|y-\hat{y}\|^2&space;&plus;&space;\lambda&space;(\Delta&space;f&plus;\nu^2f))
, where 
![eqiation](https://latex.codecogs.com/svg.image?\lambda) is the weight of the physics term.


Only 0.2 of the training datasets with noises are fed into the training process.

All of the networks are consists of 2 forword layer, with **vanilla ReLU()** activation, **PINN Sigmoid()** activation 
and **PCNN 
sin()** activation, respectively.

## Problem during training the PINN
- We found the optimization concentrated too much on the optimization of the physics term 
![equation](https://latex.codecogs.com/svg.image?\Delta&space;f&space;&plus;&space;\nu^2&space;f=0).
This will result in the prediction of 
![equation](https://latex.codecogs.com/svg.image?f) 
to be 0 if we are using the predicted 
![equation](https://latex.codecogs.com/svg.image?f) in the physical term.
While, this problem can be solved by using the 
![equation](https://latex.codecogs.com/svg.image?f) from training datasets instead of prediction.
What we want to discuss is the question that **how to avoid the NN prediction goes to 0 to 
satisfy the physics term**. 
We tried to decrease the weight of the physics term, and it works when the weight decreases to **1e-4**. Thi problem is 
mentioned in [when and why PINNs fail to train](https://www.sciencedirect.com/science/article/pii/S002199912100663X).
- Trained ont he with the datasets generated with the same Bessel kernel where 
![equation](https://latex.codecogs.com/svg.image?k=2), the PINN can have 
different value of 
![equation](https://latex.codecogs.com/svg.image?\nu) which should be the converged at a same 
value except the value of 
![equation](https://latex.codecogs.com/svg.image?k) changed. 
After checking we found the value of ![equation](https://latex.codecogs.com/svg.image?\nu) is 
 diverge from 
![equation](https://latex.codecogs.com/svg.image?k=2), most likely to be around 2.0.
**There maybe something wrong with the code or the theory.** 

    
| ![space-1.jpg](./helmholtz_2d/xy_data/contourf_helmholtz_noise.png) | ![space-1.jpg](./helmholtz_2d/xy_data/contourf_helmholtz_Truth.png) |
|:--:| :--:| 
| **Training datasets with noise (2-D)** |**Training datasets without noise (2-D)**|
| ![space-1.jpg](./helmholtz_2d/xy_data/training_loss.png) | ![space-1.jpg](./helmholtz_2d/xy_data/vanilla.png) |
| **Training loss (2-D)** |**Vanilla prediction (2-D)**|
| ![space-1.jpg](./helmholtz_2d/xy_data/physics_informed.png) | ![space-1.jpg](./helmholtz_2d/xy_data/physics_constrained.png) |
| **physics_informed prediction (2-D)** |**physics_constrained prediction (2-D)**|


| ![space-1.jpg](./helmholtz_2d/xy_data/error_cut_1.00.png) | ![space-1.jpg](./helmholtz_2d/xy_data/error_cut_0.05.png) | ![](./helmholtz_2d/xy_data/error_cut_-1.00.png)|
|:--:| :--:| :--:| 
| **Cut to see the prediction (at the *top*)** |**Cut to see the prediction (at the *medium*)**|**Cut to see the prediction (at the *bottom*)**|



# Discussion
- First, only the proper activation according to the datasets can help the model regression well. For example, the
vanilla network with **ReLU()** performs poor.
- with proper the activation, the PCNN can perform as well as the PINN. 
- The PINN training is very time-consuming as the `grad` opertation in PINN trainig is fair complex. 
So it should be a better idea to apply the physics to the activation function.
- The PINN and the PCNN both can help the model generalize better and resist the noise in the training datasets as the 
physics conditions are included.








