###############
# To capture the manifold of neuronal activity in a low-dimensional space, we employed the recurrent switching linear dynamical systems (rSLDS) [1]. Code used to fit rSLDS is from the GitHub (https://github.com/lindermanlab/ssm). Firstly, we constructed a neural signal matrix which had dimensions m × n, where m represents the number of time points and n denotes the number of neurons. Then we excluded the first two minutes of the calcium signal to eliminate signal interference caused by the suspension device at the start of the experiment [2]. The key model parameters were configured as follows: the number of neural states K = 2, corresponding to the behavioral states of struggle and immobility; and the dimensionality of the continuous state D = 2 [3].


# [1] Nair A, Karigo T, Yang B, et al. An approximate line attractor in the hypothalamus encodes an aggressive state. Cell. 2023;186(1):178-193.e15. doi:10.1016/j.cell.2022.11.027
# [2] Nandi A, Virmani G, Barve A, Marathe S. DBscorer: An Open-Source Software for Automated Accurate Analysis of Rodent Behavior in Forced Swim Test and Tail Suspension Test. eNeuro. 2021;8(6):ENEURO.0305-21.2021. Published 2021 Nov 4. doi:10.1523/ENEURO.0305-21.2021
# [3] Bush NE, Ramirez JM. Latent neural population dynamics underlying breathing, opioid-induced respiratory depression and gasping. Nat Neurosci. 2024;27(2):259-271. doi:10.1038/s41593-023-01520-3
###############
import os
import pickle
import copy
import scipy.io as sio
import autograd.numpy as np
import autograd.numpy.random as npr

npr.seed(12345)

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm

import seaborn as sns

color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import pandas as pd

import ssm
from ssm.util import random_rotation, find_permutation





# Helper functions for plotting results
def plot_trajectory(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                alpha=1.0)
    return ax


def plot_observations(z, y, ax=None, ls="-", lw=1):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    color=colors[z[start] % len(colors)],
                    alpha=1.0)
    return ax


def plot_most_likely_dynamics(model,
                              xlim=(-20, 20), ylim=(-20, 20), nxpts=20, nypts=20,
                              alpha=0.8, ax=None, figsize=(3, 3)):
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        # ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax


from matplotlib import cm


def plot_most_likely_dynamics_3d(model, true_states, states_plds,
                                 xlim=(-20, 20), ylim=(-20, 20), nxpts=50, nypts=50,
                                 alpha=0.6, ax=None, figsize=(6, 6), elev=60, azim=30):
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)
    z_true = np.argmax(states_plds.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        # ax = fig.add_subplot(111)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    dxydt_m_all = np.zeros(xy.shape)
    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy
        zk = z == k
        if zk.sum(0) > 0:
            dxydt_m_all[zk, :] = dxydt_m[zk, :]

    dxydt_m_all_true = np.zeros(states_plds.shape)
    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = states_plds.dot(A.T) + b - states_plds
        zk = z_true == k
        if zk.sum(0) > 0:
            dxydt_m_all_true[zk, :] = dxydt_m[zk, :]

    dxydt_m_norm = np.zeros((len(dxydt_m_all)))
    for xt in range(0, len(dxydt_m_all)):
        dxydt_m_norm[xt] = np.linalg.norm(dxydt_m_all[xt, :])

    dxydt_m_norm_true = np.zeros((len(dxydt_m_all_true)))
    for xt in range(0, len(dxydt_m_all_true)):
        dxydt_m_norm_true[xt] = np.linalg.norm(dxydt_m_all_true[xt, :])
    dxydt_m_norm_true = dxydt_m_norm_true + 0.1

    ax.plot_trisurf(xy[:, 0], xy[:, 1], dxydt_m_norm, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False, alpha=0.8)

    dxydt_m_norm_ = np.reshape(dxydt_m_norm, X.shape, order='C')

    # ax.plot_surface(X, Y, dxydt_m_norm_, cmap=cm.coolwarm)
    # ax.contour(X, Y, dxydt_m_norm_,zdir='z',offset=-1,cmap=cm.coolwarm)
    # ax.contour(X, Y, dxydt_m_norm_,inline=False, fontsize=6)
    # ax.contour(X, Y, dxydt_m_norm_,inline=False,offset=-1,fontsize=10)

    for i in range(len(true_states)):
        if true_states[i] == 1:
            plt.plot(states_plds[i:i + 2, 0], states_plds[i:i + 2, 1], dxydt_m_norm_true[i:i + 2], '-k', lw=1, alpha=1,
                     c='r')
            continue
        else:
            plt.plot(states_plds[i:i + 2, 0], states_plds[i:i + 2, 1], dxydt_m_norm_true[i:i + 2], '-k', lw=1, alpha=1,
                     c='b')
            # continue

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$ dynamic velocity$')

    # ax.view_init(elev=60, azim=-135)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    return ax, z, dxydt_m_norm

def fit_rslds(data,label,path):
    # Global parameters
    T = data.shape[0]  # 帧数
    K = 2
    D_obs = data.shape[1]  # 细胞数
    D_latent = 2

    print(data.shape)

    # Fit an rSLDS with its default initialization, using Laplace-EM with a structured variational posterior
    rslds = ssm.SLDS(D_obs, K, D_latent,
                     transitions="recurrent_only",
                     dynamics="diagonal_gaussian",
                     emissions="gaussian_orthog",
                     single_subspace=True)
    print(rslds)

    rslds.initialize(data)


    q_elbos_lem, q_lem = rslds.fit(data, method="laplace_em",
                                   variational_posterior="structured_meanfield",
                                   initialize=False, num_iters=1000, alpha=0.0)  # default iters= 100

    xhat_lem = q_lem.mean_continuous_states[0]
    # rslds.permute(find_permutation(z, rslds.most_likely_states(xhat_lem, y)))
    zhat_lem = rslds.most_likely_states(xhat_lem, data)



    # store rslds
    rslds_lem = copy.deepcopy(rslds)

    # varience explanion
    (np.var(xhat_lem,axis = 0)/np.sum(np.var(data,axis = 0))) *100
    # Plot some results
    plt.figure()
    plt.plot(q_elbos_lem[1:], label="Laplace-EM")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.savefig(path+"ELBO.pdf")

    #可视化
    plt.figure(figsize=[10,4])
    ax= plt.subplot(133)
    plot_trajectory(zhat_lem, xhat_lem, ax=ax)
    #plt.title("Inferred, Laplace-EM")
    plt.tight_layout()
    plt.grid(False)
    plt.title("Dynamics colared by state")
    plt.savefig(path+"Dynamics colared by state.pdf")


    plt.figure(figsize=[10,4])
    ax= plt.subplot(133)
    plot_trajectory(np.squeeze(np.around(np.array(label.T))), xhat_lem, ax=ax)
    plt.tight_layout()
    plt.grid(False)
    plt.title("Dynamics colared by label")
    plt.savefig(path+"Dynamics colared by label.pdf")

    plt.figure(figsize=(10,4))
    ax = plt.subplot(132)
    lim = abs(xhat_lem).max(axis=0) + 1
    #plt.scatter(xhat_lem[:,0], xhat_lem[:,1],s=2, c='green', alpha=0.6)
    plot_most_likely_dynamics(rslds_lem, ax=ax)
    plot_trajectory(zhat_lem, xhat_lem, ax=ax)
    plt.tight_layout()
    plt.savefig(path+ "Dynamics matrix.pdf")

    #plot pie plot
    proportion_sum = np.zeros((2, 2))
    observation_label = np.squeeze(np.around(np.array(label.T)))

    index_state_0 = np.where(zhat_lem == 0)
    proportion_sum[0][0] = np.sum(observation_label[index_state_0] == 0) / len(index_state_0[0])
    proportion_sum[0][1] = np.sum(observation_label[index_state_0] == 1) / len(index_state_0[0])

    index_state_1 = np.where(zhat_lem == 1)
    proportion_sum[1][0] = np.sum(observation_label[index_state_1] == 0) / len(index_state_1[0])
    proportion_sum[1][1] = np.sum(observation_label[index_state_1] == 1) / len(index_state_1[0])
    pd.DataFrame(proportion_sum).to_csv(file_path + "Proportion_of_state.csv")

    # 子图1，显示默认属性情况，为与子图2对比，添加了外标签
    plt.figure(figsize=(6, 6))
    plt.subplot(121)
    plt.pie(proportion_sum[0], autopct='%1.1f%%')
    plt.title('State1')
    # 子图2，演示外标签相关属性
    plt.subplot(122)
    plt.pie(proportion_sum[1], autopct='%1.1f%%')
    plt.title('State2')
    plt.savefig(path + "State_pie_plot.pdf")

    #plot动力学场
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    plot_most_likely_dynamics_3d(rslds_lem, zhat_lem, xhat_lem,elev=60,azim=30)
    plt.title("Inferred Dynamics, Laplace-EM")
    plt.savefig(path+"Inferred Dynamics 1.pdf")

    plot_most_likely_dynamics_3d(rslds_lem, zhat_lem, xhat_lem,elev=60,azim=120)
    plt.title("Inferred Dynamics, Laplace-EM")
    plt.savefig(path+"Inferred Dynamics 2.pdf")

    plot_most_likely_dynamics_3d(rslds_lem, zhat_lem, xhat_lem,elev=60,azim=210)
    plt.title("Inferred Dynamics, Laplace-EM")
    plt.savefig(path+ "Inferred Dynamics 3.pdf")

    plot_most_likely_dynamics_3d(rslds_lem, zhat_lem, xhat_lem,elev=60,azim=300)
    plt.title("Inferred Dynamics, Laplace-EM")
    plt.savefig(path+ "Inferred Dynamics 4.pdf")

    attractor_scores_sum = []
    tau1_sum=[]
    tau2_sum =[]
    for i in range(K):
        # 定义动力学矩阵 W
        W = rslds_lem.dynamics.As[i]
        # 计算系统的特征值
        eigenvalues = np.linalg.eigvals(W)

        # 计算时间常数
        time_constants = 1.0 / np.abs(np.log10(np.abs(eigenvalues)))
        # 排序时间常数
        sorted_time_constants = np.sort(time_constants)[::-1]
        print("时间常数:", sorted_time_constants)

        # 计算吸引子分数
        first_tau = sorted_time_constants[0]
        second_tau = sorted_time_constants[1]
        attractor_scores = np.log2(first_tau / second_tau)
        print("吸引子分数:", attractor_scores)
        tau1_sum.append(first_tau)
        tau2_sum.append(second_tau)

        # 可视化时间常数和吸引子分数
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(sorted_time_constants)), sorted_time_constants, color='blue')
        plt.title("Time constants of Matrix")
        plt.xlabel("Dim")
        plt.ylabel("Time constants")

        plt.tight_layout()
        plt.savefig(path+"Time constants of Matrix "+str(i)+".pdf")
        attractor_scores_sum.append(attractor_scores)
    tau_sum = [tau1_sum, tau2_sum]

    # 得到两个状态下的能量值
    energy1_sum=[]
    energy2_sum = []
    ax,z,m_norm = plot_most_likely_dynamics_3d(rslds_lem, zhat_lem, xhat_lem, xlim=(min(xhat_lem[:, 0]) -5, max(xhat_lem[:, 0])+5),
                                     ylim=(min(xhat_lem[:, 1])-5, max(xhat_lem[:, 1])+5),elev=60,azim=300)
    plt.title("Inferred Dynamics, Laplace-EM")

    attractor_1_energy = min(m_norm[z==0])
    attractor_2_energy = min(m_norm[z==1])

    print("吸引子1 最低能量:",attractor_1_energy,"吸引子2 最低能量:",attractor_2_energy)
    print("最低能量差值:",attractor_2_energy-attractor_1_energy)
    energy1_sum=attractor_1_energy
    energy2_sum=attractor_2_energy
    energy_sum = [energy1_sum,energy2_sum]

    return (tau_sum,attractor_scores_sum,energy_sum)




if __name__ == "__main__":
    CAM =  [] #CAMKII neuron signal file pahtway
    PV =  []  #PV neuron signal file pahtway
    SST = []  #SST neuron signal file pahtway
    VIP =  [] #VIP neuron signal file pahtway

    CAM_tau=[]
    CAM_attractor_scores=[]
    CAM_energy=[]
    for i in CAM:
        print(i)
        file_path =  [] # project file pathway
        data = sio.loadmat(file_path+"ds_deltaff_per_cell.mat")
        data = data['de_down_sample_deltaff_per_cell']

        label = sio.loadmat(file_path+"bin_label.mat")
        label = label['de_bin_label']
        data = np.array(data)
        data = data.T

        (tau_sum,attractor_scores_sum,energy_sum) = fit_rslds(data,label,file_path)
        CAM_tau.append(tau_sum)
        CAM_attractor_scores.append(attractor_scores_sum)
        CAM_energy.append(energy_sum)

    pd.DataFrame(CAM_tau).to_csv(file_path+"CAM_tau.csv")
    pd.DataFrame(CAM_attractor_scores).to_csv(file_path + "CAM_attractor_scores.csv")
    pd.DataFrame(CAM_energy).to_csv(file_path + "CAM_energy.csv")

    PV_tau=[]
    PV_attractor_scores=[]
    PV_energy=[]
    for i in PV:
        print(i)
        file_path =  [] # project file pathway
        data = sio.loadmat(file_path+"ds_deltaff_per_cell.mat")
        data = data['de_down_sample_deltaff_per_cell']

        label = sio.loadmat(file_path+"bin_label.mat")
        label = label['de_bin_label']
        data = np.array(data)
        data = data.T

        (tau_sum,attractor_scores_sum,energy_sum) = fit_rslds(data,label,file_path)
        PV_tau.append(tau_sum)
        PV_attractor_scores.append(attractor_scores_sum)
        PV_energy.append(energy_sum)

    pd.DataFrame(PV_tau).to_csv(file_path+"PV_tau.csv")
    pd.DataFrame(PV_attractor_scores).to_csv(file_path + "PV_attractor_scores.csv")
    pd.DataFrame(PV_energy).to_csv(file_path + "PV_energy.csv")

    SST_tau=[]
    SST_attractor_scores=[]
    SST_energy=[]
    for i in SST:
        print(i)
        file_path =  [] # project file pathway
        data = sio.loadmat(file_path+"ds_deltaff_per_cell.mat")
        data = data['de_down_sample_deltaff_per_cell']

        label = sio.loadmat(file_path+"bin_label.mat")
        label = label['de_bin_label']
        data = np.array(data)
        data = data.T

        (tau_sum,attractor_scores_sum,energy_sum) = fit_rslds(data,label,file_path)
        SST_tau.append(tau_sum)
        SST_attractor_scores.append(attractor_scores_sum)
        SST_energy.append(energy_sum)

    pd.DataFrame(SST_tau).to_csv(file_path+"SST_tau.csv")
    pd.DataFrame(SST_attractor_scores).to_csv(file_path + "SST_attractor_scores.csv")
    pd.DataFrame(SST_energy).to_csv(file_path + "SST_energy.csv")

    VIP_tau=[]
    VIP_attractor_scores=[]
    VIP_energy=[]
    for i in VIP:
        print(i)
        file_path =  [] # project file pathway
        data = sio.loadmat(file_path+"ds_deltaff_per_cell.mat")
        data = data['de_down_sample_deltaff_per_cell']

        label = sio.loadmat(file_path+"bin_label.mat")
        label = label['de_bin_label']
        data = np.array(data)
        data = data.T

        (tau_sum,attractor_scores_sum,energy_sum) = fit_rslds(data,label,file_path)
        VIP_tau.append(tau_sum)
        VIP_attractor_scores.append(attractor_scores_sum)
        VIP_energy.append(energy_sum)

    pd.DataFrame(VIP_tau).to_csv(file_path+"VIP_tau.csv")
    pd.DataFrame(VIP_attractor_scores).to_csv(file_path + "VIP_attractor_scores.csv")
    pd.DataFrame(VIP_energy).to_csv(file_path + "VIP_energy.csv")