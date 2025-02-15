"""
@file generate_NASCAR.py

Handles generating a local dataset for the NASCAR switching dynamics problem
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import NASCAR.rslds.plotting as rplt

from matplotlib import gridspec
from matplotlib.font_manager import FontProperties

from scipy.special import beta
from scipy.integrate import simps
from NASCAR.pypolyagamma.binary_trees import decision_list

from NASCAR.rslds.models import PGRecurrentSLDS
from NASCAR.pybasicbayes.pybasicbayes.distributions import Regression, Gaussian, DiagonalRegression

# Constants
K_true = 4
D_latent = 2

# Parse command line arguments
parser = argparse.ArgumentParser(description='Synthetic NASCAR Example')
parser.add_argument('--T', type=int, default=10000,
                    help='number of training time steps')
parser.add_argument('--T_sim', type=int, default=2000,
                    help='number of simulation time steps')
parser.add_argument('--K', type=int, default=4,
                    help='number of inferred states')
parser.add_argument('--D_obs', type=int, default=10,
                    help='number of observed dimensions')
parser.add_argument('--mask_start', type=int, default=0,
                    help='time index of start of mask')
parser.add_argument('--mask_stop', type=int, default=0,
                    help='time index of end of mask')
parser.add_argument('--N_samples', type=int, default=1000,
                    help='number of iterations to run the Gibbs sampler')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--cache', action='store_true', default=False,
                    help='whether or not to cache the results')
parser.add_argument('-o', '--output-dir', default='.',
                    help='where to store the results')
args = parser.parse_args()

print("Setting seed to ", args.seed)
np.random.seed(args.seed)


def logistic(x):
    return 1.0 / (1 + np.exp(-x))


def get_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi) ** alpha_k * logistic(-psi) ** alpha_rest / beta(alpha_k, alpha_rest)

    return density


def compute_psi_cmoments(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10, 10, 1000)

    mu = np.zeros(K - 1)
    sigma = np.zeros(K - 1)
    for k in range(K - 1):
        density = get_density(alphas[k], alphas[k + 1:].sum())
        mu[k] = simps(psi * density(psi), psi)
        sigma[k] = simps(psi ** 2 * density(psi), psi) - mu[k] ** 2

    return mu, sigma


def simulate_nascar():
    assert K_true == 4

    def random_rotation(n, theta):
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        out = np.zeros((n, n))
        out[:2, :2] = rot
        q = np.linalg.qr(np.random.randn(n, n))[0]
        # q = np.eye(n)
        return q.dot(out).dot(q.T)

    As = [random_rotation(D_latent, np.pi / 24.),
          random_rotation(D_latent, np.pi / 48.)]

    # Set the center points for each system
    centers = [np.array([+2.0, 0.]),
               np.array([-2.0, 0.])]
    bs = [-(A - np.eye(D_latent)).dot(center) for A, center in zip(As, centers)]

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([+0.1, 0.]))

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([-0.35, 0.]))

    # Construct multinomial regression to divvy up the space #
    tree = decision_list(K_true)
    w1, b1 = np.array([+1.0, 0.0]), np.array([-2.0])  # x + b > 0 -> x > -b
    w2, b2 = np.array([-1.0, 0.0]), np.array([-2.0])  # -x + b > 0 -> x < b
    w3, b3 = np.array([0.0, +1.0]), np.array([0.0])  # y > 0

    reg_W = np.row_stack((w1, w2, w3))
    reg_b = np.row_stack((b1, b2, b3))

    # Scale the weights to make the transition boundary sharper
    reg_scale = 100.
    reg_b *= reg_scale
    reg_W *= reg_scale

    # Account for stick breaking asymmetry
    mu_b, _ = compute_psi_cmoments(np.ones(K_true))
    reg_b += mu_b[:, None]

    # Make a recurrent SLDS with these params #
    dynamics_distns = [
        Regression(
            A=np.column_stack((A, b)),
            sigma=1e-4 * np.eye(D_latent),
            nu_0=D_latent + 2,
            S_0=1e-4 * np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent + 1)),
            K_0=np.eye(D_latent + 1),
        )
        for A, b in zip(As, bs)]

    init_dynamics_distns = [
        Gaussian(
            mu=np.array([0.0, 1.0]),
            sigma=1e-3 * np.eye(D_latent))
        for _ in range(K_true)]

    C = np.hstack((np.random.randn(args.D_obs, D_latent), np.zeros((args.D_obs, 1))))
    emission_distns = \
        DiagonalRegression(args.D_obs, D_latent + 1,
                           A=C, sigmasq=1e-5 * np.ones(args.D_obs),
                           alpha_0=2.0, beta_0=2.0)

    model = PGRecurrentSLDS(
        trans_params=dict(
            A=np.hstack((np.zeros((K_true - 1, K_true)), reg_W)), b=reg_b, sigmasq_A=100., sigmasq_b=100., tree=tree),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns)

    # Sample from the model
    inputs = np.ones((args.T, 1))
    y, x, z = model.generate(T=args.T, inputs=inputs)

    # Maks off some data
    mask = np.ones((args.T, args.D_obs), dtype=bool)
    mask[args.mask_start:args.mask_stop] = False
    return model, inputs, z, x, y, mask, reg_W, reg_b


def fit_pca(y, whiten=True):
    print("Fitting PCA")
    from sklearn.decomposition import PCA
    model = PCA(n_components=D_latent, whiten=whiten)
    x_init = model.fit_transform(y)
    C_init = model.components_.T
    b_init = model.mean_[:, None]
    sigma = np.sqrt(model.explained_variance_)

    # inverse transform is given by X.dot(sigma * C_init.T) + b_init.T
    if whiten:
        C_init = sigma * C_init

    return x_init, np.column_stack((C_init, b_init))


# Plotting code
def make_figure(true_model, z_true, x_true, y):
    fig = plt.figure(figsize=(6.5, 3.5))
    gs = gridspec.GridSpec(2, 3)

    fp = FontProperties()
    fp.set_weight("bold")

    # True dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    rplt.plot_most_likely_dynamics(true_model.trans_distn,
                                   true_model.dynamics_distns,
                                   xlim=(-3, 3), ylim=(-2, 2),
                                   ax=ax1)

    # Overlay a partial trajectory
    rplt.plot_trajectory(z_true[1:1000], x_true[1:1000], ax=ax1, ls="-")
    ax1.set_title("True Latent Dynamics")
    plt.figtext(.025, 1 - .075, '(a)', fontproperties=fp)

    # Plot a few output dimensions
    ax2 = fig.add_subplot(gs[1, 0])
    for n in range(args.D_obs):
        rplt.plot_data(z_true[1:1000], y[1:1000, n], ax=ax2, ls="-")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("$y$")
    ax2.set_title("Observed Data")
    plt.figtext(.025, .5 - .075, '(b)', fontproperties=fp)

    plt.tight_layout()
    # plt.savefig(os.path.join(args.output_dir, "NASCAR.png"), dpi=200)
    plt.show()


if __name__ == '__main__':
    # Simulate NASCAR data
    true_model, inputs, z_true, x_true, y, mask, W, b = simulate_nascar()
    print(W.shape, b.shape)

    # Run PCA to get 2D dynamics
    x_init, C_init = fit_pca(y)

    make_figure(true_model, z_true, x_true, y)

    plt.plot(inputs[:1000])
    plt.show()

    plt.plot(x_init[:1000, 0])
    plt.plot(x_init[:1000, 1])
    plt.show()

    # Observation
    plt.plot(y[:1000, 0])
    plt.plot(y[:1000, 1])
    plt.plot(y[:1000, 2])
    plt.plot(y[:1000, 3])
    plt.plot(y[:1000, 4])
    plt.legend()
    plt.show()

    plt.plot(x_true[:1000, 0])
    plt.plot(x_true[:1000, 1])
    plt.show()

    plt.plot(z_true[:250])
    plt.show()

    # Reshape into sequences of 20 timesteps
    window_len = 7
    strides, labels, true_latents, ids = [], [], [], []
    for i in range(0, y.shape[0] - window_len):
        strides.append(y[i:i+window_len, :])
        labels.append(z_true[i:i+window_len])
        true_latents.append(x_true[i:i+window_len])
        ids.append(z_true[i:i+window_len])
    strides = np.stack(strides)
    labels = np.stack(labels)
    true_latents = np.stack(true_latents)
    ids = np.stack(ids)

    # Get samples and their ocntext sets
    context, query, query_label, latents, state_ids = [], [], [], [], []
    for i in range(window_len + 3, strides.shape[0] - window_len):
        context.append([
            strides[i - window_len - 2],
            strides[i - window_len - 1],
            strides[i - window_len]
        ])

        query.append(strides[i])
        query_label.append(labels[i])
        latents.append(true_latents[i])
        state_ids.append(ids[i])

    context = np.stack(context)
    query = np.stack(query)
    query_label = np.stack(query_label)
    latents = np.stack(latents)
    state_ids = np.stack(state_ids)
    
    print(f"yt_context: {context.shape}")
    print(f"yt_query: {query.shape}")
    print(f"label: {query_label.shape}")
    print(f"xt: {latents.shape}")
    print(f"state ID: {state_ids.shape}")

    np.savez("nascar_train.npz", context=context[:8000], query=query[:8000], xt=latents[:8000], params=state_ids[:8000], label=query_label[:8000])
    np.savez("nascar_test.npz", context=context[8000:], query=query[8000:],  xt=latents[8000:], params=state_ids[8000:], label=query_label[8000:])