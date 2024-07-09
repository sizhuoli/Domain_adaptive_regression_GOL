import numpy as np
import matplotlib.pyplot as plt




import seaborn as sns

from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import linregress
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import scipy


def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b)

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor=None, alpha = 0.4)

    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = scipy.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = scipy.polyfit(xs, ys + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(xs, scipy.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax

def plot_scatter(x, y, title, xlabel, ylabel, limi, spinexy = True, font = 35, markersize = 2, alpha = 1, perc = 0.95, hist = 0, xtic = 0, showr2 = 0, showfit = 1):
    # plt.rcParams['font.family'] = 'Lucida Grande'
    sns.set(style="ticks", font_scale=2)
    # sns.set_style({'font.family':'serif', 'font.serif':'Helvetica', 'font.weight':'normal'})
    x = np.array(x)
    y = np.array(y)

    def func(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(func, x, y)
    # r2 = r2_score(np.array(x), np.array(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    skr2 = r2_score(x, y)


    p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
    y_model = equation(p, x)
    n = y.size                                           # number of observations
    m = p.size                                                 # number of parameters
    dof = n - m                                                # degrees of freedom
    t = stats.t.ppf(perc, n - m)
    print(p)
    fig, ax = plt.subplots(figsize=(12, 12))
    # ax = fig.add_subplot(111)
    if spinexy:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        for key, spine in ax.spines.items():
            spine.set_visible(False)
    # Data
    # ax.plot(
    #     x, y, "o", color="#b9cfe7", markersize=8,
    #     markeredgewidth=1, markeredgecolor="b", markerfacecolor="None"
    # )
    from matplotlib import colors
    if hist:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        counts, xedges, yedges, im = ax.hist2d(x, y, bins = 50, cmap = 'Blues', density =  1, norm=colors.LogNorm(), vmin = 0.00001)
        ax2 = plt.gca()
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="2.5%", pad=0.01)
        # cbar_ax = fig.add_axes([0.9, 0.96, 0.01, 0.8])
        # fig.colorbar(thplot, cax=cbar_ax, orientation="horizontal")
        cbar=fig.colorbar(im, cax = cax)
        # cbar=fig.colorbar(im, aspect = 50)
        tick_font_size = font
        cbar.outline.set_linewidth(0.15)
        cbar.ax.tick_params(labelsize=tick_font_size)
        # plt.axis('scaled')
    else:
        ax.scatter(x, y, color = 'teal', s=markersize, alpha = 0.5)
    xx = [-30, limi]

    x2 = np.linspace(0, limi, 100)
    y2 = equation(p, x2)

    # Estimates of Error in Data/Model
    resid = y - y_model
    chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
    chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2) / dof)
    if showfit:
        plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)

    # # Prediction Interval
    # pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    # ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    # ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
    # ax.plot(x2, y2 + pi, "--", color="0.5")
    plt.xlim(-5, limi)
    plt.ylim(-5, limi)

    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    if xtic:
        plt.xticks(fontsize=font)
    else:
        plt.xticks([])
    plt.yticks(fontsize=font)
    # plt.title(title, fontsize = 16)
    if showr2:
        if showfit:
            ax.plot(np.array(xx), func(np.array(xx), *popt), 'teal', label='f(x) = %5.2f x + %5.2f\n$Pearson  r$ = %5.2f' % (popt[0], popt[1], r_value))
        else:
            ax.text(10, int(limi*0.9), '$R^2$ = %5.2f'%(skr2), size = font)
    else:
        ax.plot(np.array(xx), func(np.array(xx), *popt), 'teal', label='f(x) = %5.2f x + %5.2f' % (popt[0], popt[1]))

    ax.plot(xx, xx, '--', color = 'gray', alpha = alpha)
    # L = ax.legend(fontsize = 22,)
    # plt.setp(L.texts, family='DejaVu Sans')
    if showfit:
        ax.legend(loc = 'upper left', fontsize = font-1, handlelength=0.5)
    if xtic:
        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

    plt.xlabel(xlabel,fontsize=font)
    plt.ylabel(ylabel,fontsize=font)
    plt.tight_layout()
    plt.show()

    return


def plot_sphere(emb, label, title,
                colors = ['red', 'orange', 'green', 'blue', 'purple']):
    # plot a unit sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = np.outer(np.cos(u), np.sin(v))
    # y = np.outer(np.sin(u), np.sin(v))
    # z = np.outer(np.ones(np.size(u)), np.cos(v))
    # ax.plot_surface(x, y, z, color='gray', alpha=0.1)

    # split emb into groups according to label
    # reducec emb to 3d
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    emb = tsne.fit_transform(emb)

    emb_groups = []
    n_ranks = len(np.unique(label))
    for i in range(n_ranks):
        emb_groups.append(emb[label == i])
    # plot each group
    for i in range(n_ranks):
        # normalize so that hth = 1
        emb_groups[i] = emb_groups[i] / np.sqrt(np.sum(emb_groups[i] ** 2, axis=1, keepdims=True))
        ax.scatter(emb_groups[i][:, 0], emb_groups[i][:, 1], emb_groups[i][:, 2], color=colors[i])

    ax.set_box_aspect([1, 1, 1])

    plt.title(title)
    plt.show()



def plot_distances(right_dis, right_sim, all_other_dis, all_other_sim, adj_dis, adj_sim,
                   colors = ['red', 'orange', 'green', 'blue', 'purple'], limt = 0.5):

    # plot two subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # plot right dis and all other dis, use color and star marker for each rank
    for i in range(len(right_dis)):
        axs[0].scatter(right_dis[i], all_other_dis[i], color=colors[i], alpha=0.6, marker='*', label='rank %d' % i, s=400)

    if max(max(right_dis), max(all_other_dis), max(adj_dis)) > limt:
        limt = max(max(right_dis), max(all_other_dis), max(adj_dis)) * 2

    # plot y=x
    axs[0].plot([0, limt], [0, limt], '--', color = 'gray', label='y=x (random classifier)')
    # fill upper left with green color and lower right with red color
    axs[0].fill_between([0, limt], [0, limt], limt, color="green", alpha=0.1)
    axs[0].fill_between([0, limt], [0, limt], 0, color="red", alpha=0.1)

    # place a circle at [0,limt] denoting "best" and a cross at [limt, 0] denoting "worst"
    axs[0].scatter(0, limt, color='green', marker='o', s=100)
    axs[0].scatter(limt, 0, color='red', marker='x', s=100)

    axs[0].set_xlabel('right dis', fontsize=20)
    axs[0].set_ylabel('all other dis', fontsize=20)
    axs[0].set_title('distance to right rank vs to all other ranks in ref', fontsize=18)

    for i in range(len(right_dis)):
        axs[1].scatter(right_dis[i], adj_dis[i], color=colors[i], alpha=0.6, marker='*', label='rank %d' % i, s=400)

    axs[1].plot([0, limt], [0, limt], '--', color='gray', label='y=x (random classifier)')
    # fill upper left with green color and lower right with red color
    axs[1].fill_between([0, limt], [0, limt], limt, color="green", alpha=0.1)
    axs[1].fill_between([0, limt], [0, limt], 0, color="red", alpha=0.1)
    axs[1].scatter(0, limt, color='green', marker='o', s=100)
    axs[1].scatter(limt, 0, color='red', marker='x', s=100)
    axs[1].set_xlabel('right dis', fontsize=20)
    axs[1].set_ylabel('adj dis', fontsize=20)
    axs[1].set_title('distance to right rank vs to adjacent ranks in ref', fontsize=18)
    # set lim
    axs[0].set_xlim([0, limt])
    axs[0].set_ylim([0, limt])
    axs[1].set_xlim([0, limt])
    axs[1].set_ylim([0, limt])
    # legend at top right

    axs[0].legend(loc='upper right', fontsize=18)
    axs[1].legend(loc='upper right', fontsize=18)
    plt.show()

    plt.show()







def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot a unit sphere
    uu = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(uu), np.sin(v))
    y = np.outer(np.sin(uu), np.sin(v))
    z = np.outer(np.ones(np.size(uu)), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.1)



    # generate a random vector but gather on the sufface of the unit sphere
    # 3d vector with shape [32, 3]
    h = np.ones((32, 3))
    # randomly add noise to h
    h = h + np.random.randn(32, 3) * 0.5

    # normalize so that hth = 1
    h = h / np.sqrt(np.sum(h ** 2, axis=1, keepdims=True))

    # vector m
    m = np.ones((32, 3))*-0.5
    # randomly add noise to m
    m = m + np.random.randn(32, 3) * 0.1
    # normalize so that mth = 1
    m = m / np.sqrt(np.sum(m ** 2, axis=1, keepdims=True))


    # plot h on the unit sphere
    ax.scatter(h[:, 0], h[:, 1], h[:, 2], color = 'blue')
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], color='red')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # equal axis
    ax.set_box_aspect([1, 1, 1])

    plt.show()


def plot_emb(emb, rank, colors = ['red', 'orange',  'green', 'blue',  'purple'], dim = 2):
    fig = plt.figure(figsize=(5, 5))

    if dim == 2:
        for i in range(int(rank.max()) + 1):
            idx = np.argwhere(rank == i).flatten()
            plt.scatter(emb[idx, 0], emb[idx, 1], color=colors[i], label=str(int(i + 1)), s=10)
            plt.scatter(emb[idx, 0].mean(), emb[idx, 1].mean(), color=colors[i],
                        marker='*', s=400, edgecolors='black')
    elif dim == 1:
        for i in range(int(rank.max()) + 1):

            idx = np.argwhere(rank == i).flatten()
            y = np.zeros_like(emb[idx]) + i
            plt.scatter(emb[idx], y, color=colors[i], label=str(int(i + 1)), s=10)
            plt.scatter(emb[idx].mean(), y.mean(), color=colors[i], marker='*', s=400, edgecolors='black')


    # legend
    plt.legend()
    # no ticks
    plt.xticks([])
    plt.yticks([])
    # outline of the plot change color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')

    plt.show()



def plot_emb_labels(emb, labels, dim = 2, logscale = True, cmap = 'rainbow', save_name = None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    reducer = TSNE(n_components=dim, verbose=0, perplexity=40, n_iter=300)
    fig = plt.figure(figsize=(12, 12))
    emb = reducer.fit_transform(emb)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_box_aspect(1)
    # use labels as colors
    # nomalize labels to [0, 1] to be used as colors
    # convert label back to original scale
    if logscale:
        labels = np.exp(labels) - 1
    # use 10 and 90 percentile instead of min and max
    labels_rd = (labels - np.percentile(labels, 1)) / (np.percentile(labels, 99) - np.percentile(labels, 1))
    # cut off
    labels_rd[labels_rd < 0] = 0
    labels_rd[labels_rd > 1] = 1
    if dim == 2:
        ax.scatter(emb[:, 0], emb[:, 1], c=labels_rd, cmap=cmap, s=60)
    elif dim == 1:
        ax.scatter(emb, labels, c=labels_rd, cmap=cmap, s=60)

    # set equal axis



    # colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, extend='both', aspect=10)
    # cbar.set_label('Output $y$', labelpad=15)
    # font size

    # some tickes three values in labels
    ticks = np.linspace(0.01, 0.99, 4)
    cbar.set_ticks(ticks)

    tick_labels = np.linspace(np.percentile(labels, 1), np.percentile(labels, 99), 4)
    # use int
    if max(tick_labels) <= 1:
        # use float
        tick_labels = [round(x, 1) for x in tick_labels]
    else:
        tick_labels = [int(x) for x in tick_labels]
    cbar.set_ticklabels(tick_labels)
    # bar outline
    # font size
    cbar.ax.tick_params(labelsize=40)

    # fontsize for setlabel
    # cbar.ax.yaxis.label.set_size(34)



    # no ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    # outline of the plot change color
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')

    # plt.show()

    plt.savefig(save_name)

def plot_emb_labels_sphere(emb, labels, logscale = True, cmap = 'rainbow', save_name = None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    uu = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(uu), np.sin(v))
    y = np.outer(np.sin(uu), np.sin(v))
    z = np.outer(np.ones(np.size(uu)), np.cos(v))
    ax.plot_surface(x, y, z, color='linen', alpha=0.3)

    # theta = np.linspace(0, 2 * np.pi, 100)
    # zz = np.zeros(100)
    # xx = np.sin(theta)
    # yy = np.cos(theta)
    #
    # ax.plot(xx, yy, zz, color='gray', alpha=0.75)
    # ax.plot(zz, xx, yy, color='gray', alpha=0.75)

    # ax.set_box_aspect(1)
    # use labels as colors
    # nomalize labels to [0, 1] to be used as colors
    # convert label back to original scale
    if logscale:
        labels = np.exp(labels) - 1
    # use 10 and 90 percentile instead of min and max
    labels_rd = (labels - np.percentile(labels, 1)) / (np.percentile(labels, 99) - np.percentile(labels, 1))
    # cut off
    labels_rd[labels_rd < 0] = 0
    labels_rd[labels_rd > 1] = 1
    # plot 3d data
    # stretch to unit sphere
    emb = emb / np.sqrt(np.sum(emb ** 2, axis=1, keepdims=True))
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=labels_rd, cmap=cmap, s=20)
    # ticks off
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # axis panel off
    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])
    # ax.view_init(-10, 20)
    ax.view_init(160, 10)
    # plt.show()
    plt.savefig(save_name)
    #




def plot_embs_3d(embs, ranks):
    reducer = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
    embs = reducer.fit_transform(embs)
    # normalize to unit sphere
    embs = embs / np.sqrt(np.sum(embs ** 2, axis=1, keepdims=True))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(int(ranks.max()) + 1):
        idx = np.argwhere(ranks == i).flatten()
        ax.scatter(embs[idx, 0], embs[idx, 1], embs[idx, 2], label=str(int(i + 1)))
        ax.scatter(embs[idx, 0].mean(), embs[idx, 1].mean(), embs[idx, 2].mean(), marker='*', s=400, edgecolors='black')
    plt.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.show()








if __name__ == '__main__':
    main()