import matplotlib.pyplot as plt
import numpy as np
import os

import pylab as p
from matplotlib.font_manager import weight_dict

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z_tilde = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples
    else:
        z_tilde=None
    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    # *** END CODE HERE ***
    m,n=x.shape
    k=K
    if is_semi_supervised:
        mu=[]
        sigma=[]
        phi =np.zeros(k)
        count_labeled = np.zeros(k)
        sum_labeled = np.zeros((k,n))
        for i in range(len(z_tilde)):
            j=int(z_tilde[i,0])
            count_labeled[j]+=1
            sum_labeled[j]+=x_tilde[i,:]
        for j in range(k):
            if count_labeled[j] > 0:
                # 有标签样本存在，用它们的均值和协方差
                mu.append(sum_labeled[j] / count_labeled[j])

                # 计算该类的协方差
                points = x_tilde[z_tilde[:, 0] == j, :]
                if len(points) > 1:
                    cov = np.cov(points.T)
                else:
                    cov = np.eye(n) * 1.0  # 只有一个样本时用单位矩阵
                sigma.append(cov + np.eye(n) * 1e-6)
            else:
                # 该类别没有有标签样本，用全局统计
                mu.append(np.mean(x, axis=0))
                sigma.append(np.cov(x.T) + np.eye(n) * 1e-6)

        mu = np.array(mu)
        sigma = np.array(sigma)

        # 初始化 phi：用有标签样本的比例 + 均匀分布
        for j in range(k):
            phi[j] = (count_labeled[j] + 1) / (len(z_tilde) + k)

        # 初始化 w：每个无标签样本对每个高斯等概率
        w = np.ones((m, k)) / k
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        # *** START CODE HERE ***

        # (1) 随机初始化：将 m 个样本均匀随机分配到 K 个组
        # 为每个样本随机分配一个初始聚类
        random_assignments = np.random.choice(k, size=m)

        # (2) 初始化 mu：计算每个组的均值和协方差
        mu = []
        sigma = []
        for j in range(k):
            cluster_points = x[random_assignments == j]
            if len(cluster_points) > 0:
                mu.append(np.mean(cluster_points, axis=0))
                # 协方差矩阵加小量保证正定
                sigma.append(np.cov(cluster_points.T) + np.eye(n) * 1e-6)
            else:
                # 如果某个组没有样本，用全局均值和协方差
                mu.append(np.mean(x, axis=0))
                sigma.append(np.cov(x.T) + np.eye(n) * 1e-6)

        mu = np.array(mu)
        sigma = np.array(sigma)

        # (3) 初始化 phi：每个高斯等概率
        phi = np.ones(k) / k

        # (4) 初始化 w：每个样本对每个高斯等概率
        w = np.ones((m, k)) / k
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)
def multivariate_gaussian_pdf(x,mu,sigma):
    n=len(x)
    diff=x-mu
    sigma_reg=sigma+np.eye(n)*1e-6
    cov_det=np.linalg.det(sigma_reg)
    cov_inv=np.linalg.inv(sigma_reg)
    norm_const=1.0/(np.sqrt((2*np.pi)**n*cov_det))
    exponent=np.exp(-0.5*diff@cov_inv@diff)
    return norm_const*exponent


def run_em(x, w, phi, mu, sigma):
    eps = 1e-3
    max_iter = 1000
    m, n = x.shape
    k = len(phi)

    it = 0
    prev_ll = -np.inf

    while it < max_iter:
        # ========== E-step ==========
        p_vals = np.zeros((m, k))
        for i in range(m):
            total = 0
            for j in range(k):
                p_vals[i, j] = phi[j] * multivariate_gaussian_pdf(x[i, :], mu[j], sigma[j])
                total += p_vals[i, j]
            for j in range(k):
                w[i, j] = p_vals[i, j] / total

        # ========== M-step ==========
        # 计算 N_j
        N = np.sum(w, axis=0)

        # 更新 phi
        phi = N / m

        # 更新 mu
        mu_new = np.zeros((k, n))
        for j in range(k):
            for i in range(m):
                mu_new[j] += w[i, j] * x[i, :]
            mu_new[j] /= N[j]

        # 更新 sigma
        sigma_new = np.zeros((k, n, n))
        for j in range(k):
            for i in range(m):
                diff = x[i, :] - mu_new[j]
                sigma_new[j] += w[i, j] * np.outer(diff, diff)
            sigma_new[j] /= N[j]

        mu = mu_new
        sigma = sigma_new

        # ========== 计算对数似然 ==========
        ll = 0
        for i in range(m):
            total = 0
            for j in range(k):
                total += phi[j] * multivariate_gaussian_pdf(x[i, :], mu[j], sigma[j])
            ll += np.log(total + 1e-12)

        # ========== 检查收敛 ==========
        if it > 0 and abs(ll - prev_ll) < eps:
            print(f"Converged at iteration {it}, ll = {ll:.6f}")
            break

        if it % 10 == 0:
            print(f"Iter {it}: ll = {ll:.6f}")

        prev_ll = ll
        it += 1

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000
    m,n=x.shape
    m_tilde,n=x_tilde.shape
    k=len(phi)
    count_labeled=np.zeros(k)
    sum_labeled=np.zeros((k,n))
    for i in range(m_tilde):
        j = int(z_tilde[i, 0])
        count_labeled[j] += 1
        sum_labeled[j] += x_tilde[i, :]
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    prev_ll = -np.inf
    while it < max_iter:
        for i in range(m):
            total = 0
            for j in range(k):
                w[i,j]=phi[j] * multivariate_gaussian_pdf(x[i, :], mu[j], sigma[j])
                total +=w[i, j]
            for j in range(k):
                w[i, j] = w[i, j] / total
        N=np.sum(w, axis=0)+alpha*count_labeled
        total_N=m+alpha*m_tilde
        phi = N/total_N
        mu_new = np.zeros((k, n))
        for j in range(k):
            mu_new[j]=np.sum(w[:,j:j+1]*x, axis=0)+alpha*sum_labeled[j]
            mu_new[j] /= N[j]
        sigma_new=np.zeros((k, n, n))
        for j in range(k):
            diff=x-mu_new[j]
            weight_diff=w[:,j:j+1]*diff
            sigma_new[j]=weight_diff.T@diff
            for i in range(m_tilde):
                if int(z_tilde[i, 0]) == j:
                    diff_labeled = x_tilde[i, :] - mu_new[j]
                    sigma_new[j] += alpha * np.outer(diff_labeled, diff_labeled)
            sigma_new[j] /= N[j]
            sigma_new[j] += np.eye(n) * 1e-6
        mu = mu_new
        sigma = sigma_new
        # ========== 计算对数似然 ==========
        ll = 0

        # 无标签部分
        for i in range(m):
            total = 0
            for j in range(k):
                total += phi[j] * multivariate_gaussian_pdf(x[i, :], mu[j], sigma[j])
            ll += np.log(total + 1e-12)

        # 有标签部分（乘以 alpha）
        for i in range(m_tilde):
            j = int(z_tilde[i, 0])
            ll += alpha * np.log(phi[j] * multivariate_gaussian_pdf(x_tilde[i, :], mu[j], sigma[j]) + 1e-12)

        # 检查收敛
        if it > 0 and abs(ll - prev_ll) < eps:
            print(f"Semi-supervised converged at iteration {it}, ll = {ll:.6f}")
            break

        if it % 10 == 0:
            print(f"Semi-supervised Iter {it}: ll = {ll:.6f}")

        prev_ll = ll
        it += 1
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
