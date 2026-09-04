import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import healpy as hp
import matplotlib.ticker as ticker
import pandas as pd

def correlation_contour_plot(x,y,liminfs=None,
                             limsups=None,sigma_smoothing=2,diag_line=False,
                             xlabel='',ylabel='', Planck=False, alpha=1):
    
    x = x.ravel()
    y = y.ravel()
    xr = (0.95*np.nanmin(x), 1.05*np.nanmax(x))
    yr = (0.95*np.nanmin(y), 1.05*np.nanmax(y))
    if Planck:
        xr = (0.85*np.nanmin(x), 1.15*np.nanmax(x))
        yr = (0.85*np.nanmin(y), 1.15*np.nanmax(y))
    H, xedges, yedges = np.histogram2d(x, y, bins=200, range=[xr, yr])
    H_smooth = gaussian_filter(H, sigma=sigma_smoothing)
    P = H_smooth / H_smooth.sum()
    P_flat = P.ravel()
    idx = np.argsort(P_flat)[::-1]
    P_sorted = P_flat[idx]
    cdf = np.cumsum(P_sorted)
    levels_prob = [0.9545,0.6827, P.max()]
    levels_density = [
        P_sorted[np.searchsorted(cdf, p)]
        for p in levels_prob    
    ]
    colors_fill = [ "#91cff3",
                    "#5e9cd3"
    ]
    colors_contour = [ "#164c6c",  
                       "#164c6c"   
    ]
    if Planck:
        colors_fill = [ "#f5c09f",
                        "#e07e40"]
        colors_contour = [ "#9f4107",
                           "#9f4107"]
        
    plt.contourf(0.5 * (xedges[1:] + xedges[:-1]),0.5 * (yedges[1:] + yedges[:-1]),
    P.T,levels=levels_density,colors = colors_fill,alpha=alpha)
    plt.contour(0.5 * (xedges[1:] + xedges[:-1]),0.5 * (yedges[1:] + yedges[:-1]),P.T,
    levels=levels_density,colors = colors_contour
    )

    if liminfs is None:
        liminfs = [0.95*np.nanmin(x),0.95*np.nanmin(y)]
    if limsups is None:
        limsups = [1.05*np.nanmax(x),1.05*np.nanmax(y)]
    if diag_line:
        plt.plot([liminfs[0],limsups[0]],[liminfs[1],limsups[1]],linestyle='dashed',alpha=0.8,c='gray')

    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.xlim(liminfs[0],limsups[0])
    plt.ylim(liminfs[1],limsups[1])
    legend_patch1 = Patch(color="#5e9cd3", label=r"$68\%$")
    legend_patch2 = Patch(color="#91cff3", label=r"$95\%$")
    plt.legend(handles=[legend_patch1,legend_patch2],loc='upper left')

def compute_Cls(map_list, planck_map, lmax=10,FSKY=1):
    mean_field = np.nanmean(map_list, axis=0)
    map_list = (map_list - mean_field[np.newaxis, :])/mean_field[np.newaxis, :]
    map_list = np.nan_to_num(map_list,nan=-1.6375e+30)
    planck_map = (planck_map - mean_field)/mean_field
    planck_map = np.nan_to_num(planck_map,nan=-1.6375e+30) 
    n_maps = map_list.shape[0]
    Cls = np.zeros((n_maps, lmax+1))
    for i in range(n_maps):
        Cls[i,:] = hp.anafast(map_list[i,:], lmax=lmax)
    Cls[970,:] = np.nan
    Cls_planck = hp.anafast(planck_map, lmax=lmax)
    return Cls/FSKY, Cls_planck/FSKY

def plot_violin(Cls, Cls_planck, title, ax):
    lmax = Cls.shape[1]-1
    ell = np.arange(lmax+1)
    sns.violinplot(data=pd.DataFrame(1000*Cls), inner='quartile', cut=0, density_norm = 'width', color='lightblue',ax=ax)
    ax.scatter(ell[1:], 1000*Cls_planck[1:], color='red', marker='_', linestyle='-', s=200, linewidths=2)
    ax.legend(handles=[Line2D([0], [0], color='red', linewidth=3, linestyle='solid', label="Planck")],fontsize=20,loc='upper right')

def plot_dipole_histogram(dipole_list, planck_dipole, color, label='',bins=20,tag=None):
    plt.hist(dipole_list*1e3,histtype='step',color=color,bins=bins,density=True,alpha=0.7)
    pval = np.mean(dipole_list>=planck_dipole)
    plt.axvline(planck_dipole*1e3,color=color,label=label+' p='+str(f"{pval:.3f}"))
    if tag=='sigma': 
        plt.xlabel(r'$10^3\times C_1^{\sigma^2}$',fontsize=16)
    if tag=='tau':
        plt.xlabel(r'$10^3\times C_1^{\tau^2}$',fontsize=16)
    if tag=='gof':
        plt.xlabel(r'$10^3\times C_1^{\chi^2}$',fontsize=16)
    plt.ylabel('Probability Density')
    plt.legend()

def dipole_contour_plot(ax,x,y,Planck,liminfs=None,limsups=None,sigma_smoothing=2,diag_line=False,xlabel='',ylabel='', alpha=1):
    x = x.ravel()*1e3
    y = y.ravel()*1e3
    xr = (0, 1.05*np.nanmax(x))
    yr = (0, 1.05*np.nanmax(y))
    H, xedges, yedges = np.histogram2d(x, y, bins=100, range=[xr, yr])
    H_smooth = gaussian_filter(H, sigma=sigma_smoothing)
    P = H_smooth / H_smooth.sum()
    P_flat = P.ravel()
    idx = np.argsort(P_flat)[::-1]
    P_sorted = P_flat[idx]
    cdf = np.cumsum(P_sorted)
    levels_prob = [0.9545,0.6827, P.max()]
    levels_density = [P_sorted[np.searchsorted(cdf, p)] for p in levels_prob]
    colors_fill = [ "#91cff3","#5e9cd3"]
    colors_contour = [ "#164c6c",  "#164c6c" ]

    ax.contourf(0.5 * (xedges[1:] + xedges[:-1]),0.5 * (yedges[1:] + yedges[:-1]),
    P.T,levels=levels_density,colors = colors_fill,alpha=alpha)
    ax.contour(0.5 * (xedges[1:] + xedges[:-1]),0.5 * (yedges[1:] + yedges[:-1]),
    P.T,levels=levels_density,colors = colors_contour)
    if liminfs is None:
        liminfs = [0.95*np.nanmin(x),0.95*np.nanmin(y)]
    if limsups is None:
        limsups = [1.05*np.nanmax(x),1.05*np.nanmax(y)]
    if diag_line:
        ax.plot([liminfs[0],limsups[0]],[liminfs[1],limsups[1]],linestyle='dashed',alpha=0.8,c='gray')
    ax.axvline(Planck[0]*1e3,c='r')
    ax.axhline(Planck[1]*1e3,c='r')
    ax.scatter(Planck[0]*1e3,Planck[1]*1e3,c='r',marker='s')
    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    ax.set_xlim(liminfs[0],limsups[0])
    ax.set_ylim(liminfs[1],limsups[1])

def multiviolinplot(list_of_sim_Cls,list_of_planck_Cls,tag):
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    ax = gs.subplots(sharex=False, sharey=False)
    n = len(list_of_sim_Cls)
    plot_violin(list_of_sim_Cls[0],list_of_planck_Cls[0],title='',ax=ax[0,0])
    plot_violin(list_of_sim_Cls[1],list_of_planck_Cls[1],title='',ax=ax[0,1])
    plot_violin(list_of_sim_Cls[2],list_of_planck_Cls[2],title='',ax=ax[1,0])
    if n>3:
        plot_violin(list_of_sim_Cls[3],list_of_planck_Cls[3],title='',ax=ax[1,1])
    for i,a in enumerate(ax.flat):
        if i<n:
            a.set_xlim(left=0.5)
            a.set_ylim([0,975*np.nanmax(list_of_sim_Cls[i][:,1])])
            a.tick_params(axis='both', which='major', labelsize=18)
        else:
            a.axis('off')
    for a in ax[:, 1]:
        a.yaxis.tick_right()
        a.yaxis.set_label_position("right")
    for a in ax[1, :]:
        a.set_xlabel(r'$\ell$',fontsize=24)
    for a in ax[:, 0]:
        a.set_ylabel(r'$10^3\times C_{\ell}$',fontsize=24)
    if tag=="sigma":
        titles = [r'$\sigma^2_{patch}$', r'$\sigma^2_{V0}$', r'$\sigma^2_{V1}$', r'$\sigma^2_{V2}$']
    if tag=="tau":
        titles = [r'$\tau^2_{patch}$', r'$\tau^2_{V1}$', r'$\tau^2_{V2}$']
    if tag=="gof":
        titles = [r'$\chi^2_{V0}$', r'$\chi^2_{V1}$', r'$\chi^2_{V2}$'] 
    for i, a in enumerate(ax.flat):
        if i < n:
            a.text(
                0.25, 0.95, titles[i],
                transform=a.transAxes,
                fontsize=24,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
def bigmultiviolinplot(list_of_sim_Cls,list_of_planck_Cls):
    list_of_sim_Cls = np.array(list_of_sim_Cls)
    list_of_planck_Cls = np.array(list_of_planck_Cls)
    fig = plt.figure(figsize=(12,12))
    gs = fig.add_gridspec(3, 4, hspace=0, wspace=0)
    ax = gs.subplots(sharex=False, sharey=False)
    plot_violin(list_of_sim_Cls[0],list_of_planck_Cls[0],title='',ax=ax[0,0])
    plot_violin(list_of_sim_Cls[1],list_of_planck_Cls[1],title='',ax=ax[0,1])
    plot_violin(list_of_sim_Cls[2],list_of_planck_Cls[2],title='',ax=ax[0,2])
    plot_violin(list_of_sim_Cls[3],list_of_planck_Cls[3],title='',ax=ax[0,3])
    plot_violin(list_of_sim_Cls[4],list_of_planck_Cls[4],title='',ax=ax[1,0])
    plot_violin(list_of_sim_Cls[5],list_of_planck_Cls[5],title='',ax=ax[1,2])
    plot_violin(list_of_sim_Cls[6],list_of_planck_Cls[6],title='',ax=ax[1,3])
    plot_violin(list_of_sim_Cls[7],list_of_planck_Cls[7],title='',ax=ax[2,1])
    plot_violin(list_of_sim_Cls[8],list_of_planck_Cls[8],title='',ax=ax[2,2])
    plot_violin(list_of_sim_Cls[9],list_of_planck_Cls[9],title='',ax=ax[2,3])
    ind_list = [[0,1],[0,2],[1,1],[1,2]]
    for [i,j] in ind_list:
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
    ax[2,2].set_yticks([])
    for a in ax[0,:]:
        a.set_xlim(left=0.5)
        a.set_ylim([0,1300*np.nanmax(list_of_planck_Cls[:4][:,1])])
    for a in ax[1,:]:
        a.set_xlim(left=0.5)
        a.set_ylim([0,1300*np.nanmax(list_of_planck_Cls[4:7][:,1])])
    for a in ax[2,:]:
        a.set_xlim(left=0.5)
        a.set_ylim([0,1350*np.nanmax(list_of_planck_Cls[7:][:,1])]) 
    ax[1,1].axis('off')
    ax[2,0].axis('off') 
    for a in ax[:, 3]:
        a.yaxis.tick_right()
    for a in ax[2, :]:
        a.set_xlabel(r'$\ell$',fontsize=14)
    ax[1,0].set_xlabel(r'$\ell$',fontsize=14)
    for a in ax[:, 0]:
        a.set_ylabel(r'$10^3\times C_{\ell}$',fontsize=14)
    ax[2,1].set_ylabel(r'$10^3\times C_{\ell}$',fontsize=14)
    ax[2,0].legend(handles=[Line2D([0], [0], color='red', linewidth=2, linestyle='solid', label="Planck"),
                            Patch(facecolor='lightblue', edgecolor='gray', label='FFP10 Simulations')],
                            fontsize=12,loc='lower left')
    fs = 18
    box_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.0)
    ax[0, 0].text(-0.4, 0.5, r'$\sigma^2 \vphantom{\Big|}$', transform=ax[0, 0].transAxes, 
                  fontsize=fs, va='center', ha='center', bbox=box_props)
    ax[1, 0].text(-0.4, 0.5, r'$\tau^2 \rule{0pt}{5em}$', transform=ax[1, 0].transAxes, 
                  fontsize=fs, va='center', ha='center', bbox=box_props)
    ax[2, 0].text(-0.4, 0.5, r'$\chi^2 \rule{0pt}{2em}$', transform=ax[2, 0].transAxes, 
                  fontsize=fs, va='center', ha='center', bbox=box_props)
    ax[0, 0].set_title('Patch', bbox=box_props,fontsize=fs,pad=15)
    ax[0, 1].set_title('V0', bbox=box_props,fontsize=fs,pad=15)
    ax[0, 2].set_title('V1', bbox=box_props,fontsize=fs,pad=15)
    ax[0, 3].set_title('V2', bbox=box_props,fontsize=fs,pad=15)
import matplotlib.ticker as ticker

def multiamplitude(amplitude_list1, amplitude_list2,amplitude_list3,
    planck_amplitude1, planck_amplitude2, planck_amplitude3,
    my_labels, title,limsups):
    fig, axes = plt.subplots(2, 2, figsize=(7, 7),sharex=False,sharey=False)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    dipole_contour_plot(axes[0,0], amplitude_list1,amplitude_list2,[planck_amplitude1,planck_amplitude2],
                        sigma_smoothing=5,ylabel=my_labels[1],limsups=limsups[0])
    dipole_contour_plot(axes[1,0], amplitude_list1,amplitude_list3,[planck_amplitude1,planck_amplitude3],
                         sigma_smoothing=5,xlabel=my_labels[0],ylabel=my_labels[2],limsups=limsups[1])
    dipole_contour_plot(axes[1,1], amplitude_list2,amplitude_list3,[planck_amplitude2,planck_amplitude3],
                         sigma_smoothing=5,xlabel=my_labels[1],limsups=limsups[2])
    axes[0, 1].axis('off')
    axes[1,1].yaxis.set_visible(False)
    from matplotlib.patches import Patch
    legend_patch1 = Patch(color="#5e9cd3", label=r"$68\%$")
    legend_patch2 = Patch(color="#91cff3", label=r"$95\%$")
    legend_line = Line2D([0], [0], color='red', linewidth=2, linestyle='solid', label="Planck")
    handles=[legend_patch1,legend_patch2, legend_line]
    axes[0, 1].legend(handles=handles, loc='upper right', frameon=True,fontsize=16)
    y_lim_00 = axes[0,0].get_ylim()
    x_lim_11 = axes[1,1].get_xlim()
    common_min = min(y_lim_00[0], x_lim_11[0])
    common_max = max(y_lim_00[1], x_lim_11[1])
    axes[0,0].set_ylim(common_min, common_max)
    axes[1,1].set_xlim(common_min, common_max)
    shared_locator = ticker.MaxNLocator(nbins=5, prune='lower')
    axes[0,0].yaxis.set_major_locator(shared_locator)
    axes[1,1].xaxis.set_major_locator(shared_locator)
    axes[0,0].set_xlim(axes[1,0].get_xlim())
    axes[1,1].set_ylim(axes[1,0].get_ylim())
    plt.subplots_adjust(wspace=0, hspace=0)

def multisigmatau(sigmalist, taulist,psigma,ptau,
    my_labels, title, liminfs,limsups,turnoff_planck=False):
    fig, axes = plt.subplots(3, 1, figsize=(4, 12),sharex=False,sharey=False)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    dipole_contour_plot(axes[0], sigmalist[0],taulist[0],[psigma[0],ptau[0]],
                        sigma_smoothing=8,xlabel=my_labels[0],ylabel=my_labels[1],limsups=limsups[0],liminfs=liminfs[0])
    dipole_contour_plot(axes[1], sigmalist[1],taulist[1],[psigma[1],ptau[1]],
                         sigma_smoothing=8,xlabel=my_labels[0],ylabel=my_labels[1],limsups=limsups[1],liminfs=liminfs[1])
    dipole_contour_plot(axes[2], sigmalist[2],taulist[2],[psigma[2],ptau[2]],
                         sigma_smoothing=8,xlabel=my_labels[0],ylabel=my_labels[1],limsups=limsups[2],liminfs=liminfs[2])
    from matplotlib.patches import Patch
    legend_patch1 = Patch(color="#5e9cd3", label=r"$68\%$")
    legend_patch2 = Patch(color="#91cff3", label=r"$95\%$")
    legend_line = Line2D([0], [0], color='red', linewidth=2, linestyle='solid', label="Planck")
    if turnoff_planck:
        handles=[legend_patch1,legend_patch2]
    else:
        handles=[legend_patch1,legend_patch2,legend_line]
    axes[0].legend(handles=handles, loc='upper right', frameon=True,fontsize=12)
    axes[0].text(
                0.1, 0.95, r'Patch',
                transform=axes[0].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    axes[1].text(
                0.1, 0.95, r'V1',
                transform=axes[1].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    axes[2].text(
                0.1, 0.95, r'V2',
                transform=axes[2].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    plt.subplots_adjust(wspace=0, hspace=0.25)

def multisigmatau2(sigmalist, taulist,psigma,ptau,
    my_labels, title, liminfs,limsups,turnoff_planck=False):
    fig, axes = plt.subplots(3, 1, figsize=(4, 12),sharex=True,sharey=False)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    dipole_contour_plot(axes[0], sigmalist[0],taulist[0],[psigma[0],ptau[0]],
                        sigma_smoothing=6,xlabel='',ylabel=my_labels[1],limsups=limsups[0],liminfs=liminfs[0])
    dipole_contour_plot(axes[1], sigmalist[1],taulist[1],[psigma[1],ptau[1]],
                         sigma_smoothing=6,xlabel='',ylabel=my_labels[1],limsups=limsups[1],liminfs=liminfs[1])
    dipole_contour_plot(axes[2], sigmalist[2],taulist[2],[psigma[2],ptau[2]],
                         sigma_smoothing=6,xlabel=my_labels[0],ylabel=my_labels[1],limsups=limsups[2],liminfs=liminfs[2])
    from matplotlib.patches import Patch
    legend_patch1 = Patch(color="#5e9cd3", label=r"$68\%$")
    legend_patch2 = Patch(color="#91cff3", label=r"$95\%$")
    legend_line = Line2D([0], [0], color='red', linewidth=2, linestyle='solid', label="Planck")
    if turnoff_planck:
        handles=[legend_patch1,legend_patch2]
    else:
        handles=[legend_patch1,legend_patch2,legend_line]
    axes[0].legend(handles=handles, loc='upper right', frameon=True,fontsize=12)
    axes[0].text(
                0.1, 0.95, r'Patch',
                transform=axes[0].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    axes[1].text(
                0.1, 0.95, r'V1',
                transform=axes[1].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    axes[2].text(
                0.1, 0.95, r'V2',
                transform=axes[2].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    plt.subplots_adjust(wspace=0, hspace=0)

def multisigmatau_row(sigmalist, taulist, psigma, ptau,
    my_labels, title, liminfs, limsups, turnoff_planck=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    dipole_contour_plot(axes[0], sigmalist[0],taulist[0],[psigma[0],ptau[0]],
                        sigma_smoothing=6,xlabel=my_labels[0],ylabel=my_labels[1],limsups=limsups[0],liminfs=liminfs[0])
    dipole_contour_plot(axes[1], sigmalist[1],taulist[1],[psigma[1],ptau[1]],
                         sigma_smoothing=6,xlabel=my_labels[0],ylabel='',limsups=limsups[1],liminfs=liminfs[1])
    dipole_contour_plot(axes[2], sigmalist[2],taulist[2],[psigma[2],ptau[2]],
                         sigma_smoothing=6,xlabel=my_labels[0],ylabel='',limsups=limsups[2],liminfs=liminfs[2])
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_patch1 = Patch(color="#5e9cd3", label=r"$68\%$")
    legend_patch2 = Patch(color="#91cff3", label=r"$95\%$")
    legend_line = Line2D([0], [0], color='red', linewidth=2, linestyle='solid', label="Planck")
    if turnoff_planck:
        handles=[legend_patch1,legend_patch2]
    else:
        handles=[legend_patch1,legend_patch2,legend_line]
    axes[0].legend(handles=handles, loc='upper right', frameon=True, fontsize=12)
    axes[0].text(
                0.1, 0.95, r'Patch',
                transform=axes[0].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    axes[1].text(
                0.1, 0.95, r'V1',
                transform=axes[1].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    axes[2].text(
                0.1, 0.95, r'V2',
                transform=axes[2].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                )
            )
    import matplotlib.ticker as ticker
    shared_locator = ticker.MaxNLocator(nbins=5)
    axes[0].xaxis.set_major_locator(shared_locator)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.canvas.draw()
    for ax in axes[1:]:
        labels = ax.get_xticklabels()
        if labels:
            labels[0].set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)

def new_dipole_contour_plot(ax, x, y, Planck, liminfs, limsups, sigma_smoothing=2, alpha=1):
    x = x.ravel() * 1e3
    y = y.ravel() * 1e3
    Planck_x = Planck[0] * 1e3
    Planck_y = Planck[1] * 1e3
    xr = (0, 1.05 * np.nanmax(x))
    yr = (0, 1.05 * np.nanmax(y))
    H, xedges, yedges = np.histogram2d(x, y, bins=100, range=[xr, yr])
    H_smooth = gaussian_filter(H, sigma=sigma_smoothing)
    P = H_smooth / H_smooth.sum()
    P_flat = P.ravel()
    idx = np.argsort(P_flat)[::-1]
    P_sorted = P_flat[idx]
    cdf = np.cumsum(P_sorted)
    levels_prob = [0.9545, 0.6827, P.max()]
    levels_density = [P_sorted[np.searchsorted(cdf, p)] for p in levels_prob]
    colors_fill = ["#91cff3", "#5e9cd3"]
    colors_contour = ["#164c6c", "#164c6c"]
    ax.contourf(
        0.5 * (xedges[1:] + xedges[:-1]),
        0.5 * (yedges[1:] + yedges[:-1]),
        P.T,
        levels=levels_density,
        colors=colors_fill,
        alpha=alpha
    )
    ax.contour(
        0.5 * (xedges[1:] + xedges[:-1]),
        0.5 * (yedges[1:] + yedges[:-1]),
        P.T,
        levels=levels_density,
        colors=colors_contour
    )
    ax.axvline(Planck_x, c='r', linewidth=2)
    ax.axhline(Planck_y, c='r', linewidth=2)
    ax.scatter(Planck_x, Planck_y, c='r', marker='s', s=40, zorder=5)
    ax.set_xlim(liminfs[0], limsups[0])
    ax.set_ylim(liminfs[1], limsups[1])

def new_corner_contour_plot(amplitude_lists, planck_amplitudes, my_labels, title, figsize=(24, 24), group_sizes=None):
    N = len(amplitude_lists)
    fig, axes = plt.subplots(N, N, figsize=figsize)
    fs_label = 18
    fs_ticks = 20
    fig.suptitle(title, fontsize=26, fontweight='bold', y=0.92)
    min_vals = []
    max_vals = []
    for i in range(N):
        data_min = np.nanmin(amplitude_lists[i]) * 1e3
        data_max = np.nanmax(amplitude_lists[i]) * 1e3
        p_val = planck_amplitudes[i] * 1e3
        min_vals.append(0.0)
        max_vals.append(p_val * 1.5)
    for row in range(N):
        for col in range(N):
            ax = axes[row, col]
            if col < row:
                new_dipole_contour_plot(
                    ax, 
                    amplitude_lists[col], amplitude_lists[row], 
                    [planck_amplitudes[col], planck_amplitudes[row]], 
                    liminfs=[min_vals[col], min_vals[row]],
                    limsups=[max_vals[col], max_vals[row]],
                    sigma_smoothing=7
                )
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='lower'))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='lower'))
                ax.tick_params(axis='both', labelsize=fs_ticks)
                if col == 0:
                    continue
                else:
                    ax.tick_params(axis='y', labelleft=False)
                if row == N - 1:
                    continue
                else:
                    ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.patch.set_alpha(0.0)
                if col == row:
                    ax.text(0.5, 0.5, my_labels[col], transform=ax.transAxes, 
                            fontsize=50, fontweight='bold', ha='center', va='center')
    axes[9,4].set_xlabel(r'$10^3\times C_1$',fontsize=40)
    axes[5,0].set_ylabel(r'$10^3\times C_1$',fontsize=40)
    legend_patch1 = Patch(color="#5e9cd3", label=r"$68\%$")
    legend_patch2 = Patch(color="#91cff3", label=r"$95\%$")
    legend_line = Line2D([0], [0], color='red', linewidth=2.5, linestyle='solid', label="Planck")
    fig.legend(handles=[legend_patch1, legend_patch2, legend_line], 
               loc='upper right', bbox_to_anchor=(0.85, 0.85), fontsize=30, frameon=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    if group_sizes:
        splits = (np.cumsum(group_sizes)[:-1] - 1).tolist()
        lw_thick = 4.5  
        for split in splits:
            for col in range(split + 1):
                ax_top = axes[split, col]
                ax_bot = axes[split + 1, col]
                ax_top.spines['bottom'].set_linewidth(lw_thick)
                ax_top.spines['bottom'].set_visible(True)
                ax_top.spines['bottom'].set_clip_on(False)
                ax_top.spines['bottom'].set_zorder(100)
                ax_bot.spines['top'].set_linewidth(lw_thick)
                ax_bot.spines['top'].set_visible(True)
                ax_bot.spines['top'].set_clip_on(False)
                ax_bot.spines['top'].set_zorder(100)
            for row in range(split + 1, N):
                ax_left = axes[row, split]
                ax_right = axes[row, split + 1]
                ax_left.spines['right'].set_linewidth(lw_thick)
                ax_left.spines['right'].set_visible(True)
                ax_left.spines['right'].set_clip_on(False)
                ax_left.spines['right'].set_zorder(100)
                ax_right.spines['left'].set_linewidth(lw_thick)
                ax_right.spines['left'].set_visible(True)
                ax_right.spines['left'].set_clip_on(False)
                ax_right.spines['left'].set_zorder(100)
    return fig, axes

def new_corner_contour_plot(amplitude_lists, planck_amplitudes, my_labels, title, figsize=(24, 24), group_sizes=None):
    N = len(amplitude_lists)
    fig, axes = plt.subplots(N, N, figsize=figsize)
    fs_label = 18
    fs_ticks = 20
    fig.suptitle(title, fontsize=26, fontweight='bold', y=0.92)
    min_vals = [0.0] * N
    max_vals = [0.0] * N
    if group_sizes is not None:
        current_idx = 0
        for size in group_sizes:
            group_indices = list(range(current_idx, current_idx + size))
            group_max_extent = 0.0
            for idx in group_indices:
                data_extent = np.nanpercentile(amplitude_lists[idx] * 1e3, 95)
                p_val = planck_amplitudes[idx] * 1e3
                var_max = max(data_extent, p_val)
                if var_max > group_max_extent:
                    group_max_extent = var_max
            shared_max = 1.2 * group_max_extent
            for idx in group_indices:
                max_vals[idx] = shared_max
            current_idx += size
    else:
        for i in range(N):
            data_extent = np.nanpercentile(amplitude_lists[i] * 1e3, 95)
            p_val = planck_amplitudes[i] * 1e3
            max_vals[i] = 1.2 * max(data_extent, p_val)
    for row in range(N):
        for col in range(N):
            ax = axes[row, col]
            if col < row:
                new_dipole_contour_plot(
                    ax, 
                    amplitude_lists[col], amplitude_lists[row], 
                    [planck_amplitudes[col], planck_amplitudes[row]], 
                    liminfs=[min_vals[col], min_vals[row]],
                    limsups=[max_vals[col], max_vals[row]],
                    sigma_smoothing=7
                )
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='lower'))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='lower'))
                ax.tick_params(axis='both', labelsize=fs_ticks)
                if col == 0:
                    continue
                else:
                    ax.tick_params(axis='y', labelleft=False)
                if row == N - 1:
                    continue
                else:
                    ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.patch.set_alpha(0.0)
                if col == row:
                    ax.text(0.5, 0.5, my_labels[col], transform=ax.transAxes, 
                            fontsize=50, fontweight='bold', ha='center', va='center')
    axes[9,4].set_xlabel(r'$10^3\times C_1$',fontsize=40)
    axes[5,0].set_ylabel(r'$10^3\times C_1$',fontsize=40)
    legend_patch1 = Patch(color="#5e9cd3", label=r"$68\%$")
    legend_patch2 = Patch(color="#91cff3", label=r"$95\%$")
    legend_line = Line2D([0], [0], color='red', linewidth=2.5, linestyle='solid', label="Planck")
    fig.legend(handles=[legend_patch1, legend_patch2, legend_line], 
               loc='upper right', bbox_to_anchor=(0.85, 0.85), fontsize=30, frameon=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    if group_sizes:
        splits = (np.cumsum(group_sizes)[:-1] - 1).tolist()
        lw_thick = 4.5  
        for split in splits:
            for col in range(split + 1):
                ax_top = axes[split, col]
                ax_bot = axes[split + 1, col]
                ax_top.spines['bottom'].set_linewidth(lw_thick)
                ax_top.spines['bottom'].set_visible(True)
                ax_top.spines['bottom'].set_clip_on(False)
                ax_top.spines['bottom'].set_zorder(100)
                ax_bot.spines['top'].set_linewidth(lw_thick)
                ax_bot.spines['top'].set_visible(True)
                ax_bot.spines['top'].set_clip_on(False)
                ax_bot.spines['top'].set_zorder(100)
            for row in range(split + 1, N):
                ax_left = axes[row, split]
                ax_right = axes[row, split + 1]
                ax_left.spines['right'].set_linewidth(lw_thick)
                ax_left.spines['right'].set_visible(True)
                ax_left.spines['right'].set_clip_on(False)
                ax_left.spines['right'].set_zorder(100)
                ax_right.spines['left'].set_linewidth(lw_thick)
                ax_right.spines['left'].set_visible(True)
                ax_right.spines['left'].set_clip_on(False)
                ax_right.spines['left'].set_zorder(100)
    return fig, axes

def extract_dipole_dirs(map_list,planck_map):
    mean_field = np.nanmean(map_list, axis=0)
    map_list = (map_list - mean_field[np.newaxis, :])/mean_field[np.newaxis, :]
    map_list = np.nan_to_num(map_list,nan=-1.6375e+30)
    planck_map = (planck_map - mean_field)/mean_field
    planck_map = np.nan_to_num(planck_map,nan=-1.6375e+30)
    dirs = np.zeros((1000,3))
    dirs[970,:] = np.nan
    for i in range(1000):
        if i != 970:
            dirs[i,:]=hp.remove_dipole(map_list[i,:],fitval=True)[2]
    planck_dir = hp.remove_dipole(planck_map,fitval=True)[2]
    return dirs/np.linalg.norm(dirs,axis=1)[:,np.newaxis], planck_dir/np.linalg.norm(planck_dir)

def setup_scientific_notation(ax, hide=False):
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.yaxis.get_major_formatter().set_useMathText(True)
    if hide:
        ax.yaxis.get_offset_text().set_visible(False)

def cosine_histogram_ax(ax,dip1,dip2,dip_planck1, dip_planck2,hide=False,fs_p=16):
    dot_products = np.einsum('ij,ij->i', dip1, dip2)
    planck_dot_product = np.dot(dip_planck1, dip_planck2)
    p_val = np.sum(dot_products >= planck_dot_product) / len(dot_products)
    counts, _ = np.histogram(dot_products, bins=30, range=(-1, 1))
    max_count = counts.max()
    weights = np.ones_like(dot_products) / max_count
    ax.hist(dot_products, bins=30, range=(-1, 1), weights=weights, 
            density=False, label='Simulations', color='tab:blue')
    ax.axvline(planck_dot_product, color='red', linestyle='--', linewidth=2.5, label='Planck')
    ax.text(0.08, 0.90, f'$p={p_val:.3f}$', 
            transform=ax.transAxes, 
            fontsize=fs_p, 
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, ec='red'))
    setup_scientific_notation(ax,hide)


def corner_cosine_histogram(
    dipole_list1, dipole_list2,dipole_list3,
    planck_dipole1, planck_dipole2, planck_dipole3,
    my_labels, title
):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5),sharex=True,sharey=True)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    cosine_histogram_ax(axes[0, 0], dipole_list1, dipole_list2, planck_dipole1, planck_dipole2)
    cosine_histogram_ax(axes[1, 0], dipole_list1, dipole_list3, planck_dipole1, planck_dipole3,hide=True)
    cosine_histogram_ax(axes[1, 1], dipole_list2, dipole_list3, planck_dipole2, planck_dipole3)
    axes[0, 1].axis('off')
    fs = 14 
    ylabel_kwargs = {'fontsize': fs, 'rotation': 0, 'labelpad': 18, 'y': 0.43}
    box_props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', linewidth=1.0)
    axes[0, 0].set_ylabel(my_labels[1],bbox=box_props, **ylabel_kwargs)
    axes[0, 0].yaxis.set_label_position("left")
    axes[0, 0].set_title(my_labels[0], bbox=box_props,fontsize=fs,pad=15)
    axes[1, 0].set_ylabel(my_labels[2],bbox=box_props, **ylabel_kwargs)
    axes[1, 1].set_title(my_labels[1],bbox=box_props, fontsize=fs,pad=15)
    axes[1,0].set_xlabel(r'$\cos(\alpha_{ij})$',fontsize=12)
    axes[1,1].set_xlabel(r'$\cos(\alpha_{ij})$',fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc='upper right', frameon=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    axes[0, 0].tick_params(axis='x', which='both', bottom=False)
    axes[1, 1].tick_params(axis='y', which='both', left=False)

def new_corner_cosine_histogram(dipole_lists, planck_dipoles, my_labels, title, figsize=(24, 24), group_sizes=None):
    N = len(dipole_lists)
    fig, axes = plt.subplots(N, N, figsize=figsize)
    fs_label = 28
    fs_ticks = 18
    fs_p = 20
    fig.suptitle(title, fontsize=26, fontweight='bold', y=0.92)
    box_props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.5)

    for row in range(N):
        for col in range(N):
            ax = axes[row, col]
            if col < row:
                cosine_histogram_ax(
                    ax, dipole_lists[col], dipole_lists[row], 
                    planck_dipoles[col], planck_dipoles[row], 
                    fs_p=fs_p
                )
                ax.set_xlim(-1.1, 1.1)
                ax.set_xticks([-1, 0, 1])
                ax.set_ylim(0, 1.15) 
                ax.set_yticks([0, 0.5, 1.0])
                ax.tick_params(axis='both', labelsize=fs_ticks)

                if col == 0:
                    abc=1
                else:
                    ax.tick_params(axis='y', labelleft=False)

                if row == N - 1:
                    ax.set_xlabel(r'$\cos(\alpha_{ij})$', fontsize=fs_label, labelpad=10)
                else:
                    ax.tick_params(axis='x', labelbottom=False)
                    
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.patch.set_alpha(0.0)
                if col == row:
                    ax.text(0.5, 0.5, my_labels[col], transform=ax.transAxes, 
                            fontsize=40, fontweight='bold', ha='center', va='center') #bbox=box_props

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.85, 0.85), fontsize=30, frameon=True)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    if group_sizes:
        splits = (np.cumsum(group_sizes)[:-1] - 1).tolist()
        lw_thick = 6 
        for split in splits:
            for col in range(split + 1):
                ax_top = axes[split, col]
                ax_bot = axes[split + 1, col]
                
                ax_top.spines['bottom'].set_linewidth(lw_thick)
                ax_top.spines['bottom'].set_visible(True)
                ax_top.spines['bottom'].set_clip_on(False)
                ax_top.spines['bottom'].set_zorder(100)
                
                ax_bot.spines['top'].set_linewidth(lw_thick)
                ax_bot.spines['top'].set_visible(True)
                ax_bot.spines['top'].set_clip_on(False)
                ax_bot.spines['top'].set_zorder(100)
                
            for row in range(split + 1, N):
                ax_left = axes[row, split]
                ax_right = axes[row, split + 1]
                
                ax_left.spines['right'].set_linewidth(lw_thick)
                ax_left.spines['right'].set_visible(True)
                ax_left.spines['right'].set_clip_on(False)
                ax_left.spines['right'].set_zorder(100)
                
                ax_right.spines['left'].set_linewidth(lw_thick)
                ax_right.spines['left'].set_visible(True)
                ax_right.spines['left'].set_clip_on(False)
                ax_right.spines['left'].set_zorder(100)

    return fig, axes

def plot_dipole_density_subplot(dipole_vectors, sub_index, title, nside=64, fwhm_deg=15.0):
    vectors = np.atleast_2d(np.array(dipole_vectors))
    if vectors.shape[1] != 3 and vectors.shape[0] == 3:
        vectors = vectors.T
    mask_finite = np.isfinite(vectors).all(axis=1)
    norms = np.linalg.norm(vectors, axis=1)
    mask_nonzero = norms > 1e-9
    clean_vectors = vectors[mask_finite & mask_nonzero].T 
    if clean_vectors.shape[1] == 0:
        return 
    npix = hp.nside2npix(nside)
    density_map = np.zeros(npix)
    theta, phi = hp.vec2ang(clean_vectors)
    pixel_indices = hp.ang2pix(nside, theta, phi)
    np.add.at(density_map, pixel_indices, 1)
    smoothed_map = hp.smoothing(density_map, fwhm=np.radians(fwhm_deg), verbose=False)
    if np.max(smoothed_map) > 0:
        smoothed_map /= np.max(smoothed_map)
    hp.mollview(
        smoothed_map, 
        title=title, 
        cmap='plasma',
        cbar=False,       
        sub=sub_index,
        min=0, max=1,
        margins=(0.01, 0.01, 0.01, 0.01)
    )
    hp.graticule(dpar=30, dmer=30, alpha=0.3, verbose=False)
    
def plot_dipole_density(dipole_vectors, nside=64, fwhm_deg=15.0, title="Dipole Density Heatmap"):
    vectors = np.array(dipole_vectors)
    mask_finite = np.isfinite(vectors).all(axis=1)
    norms = np.linalg.norm(vectors, axis=1)
    mask_nonzero = norms > 1e-9
    valid_mask = mask_finite & mask_nonzero
    clean_vectors = vectors[valid_mask].T
    n_dropped = len(vectors) - clean_vectors.shape[1]
    if n_dropped > 0:
        print(f"Warning: Dropped {n_dropped} invalid/NaN vectors.")
    if clean_vectors.shape[1] == 0:
        print("Error: No valid vectors left to plot.")
        return
    npix = hp.nside2npix(nside)
    density_map = np.zeros(npix)
    theta, phi = hp.vec2ang(clean_vectors)
    pixel_indices = hp.ang2pix(nside, theta, phi)
    np.add.at(density_map, pixel_indices, 1)
    fwhm_rad = np.radians(fwhm_deg)
    smoothed_map = hp.smoothing(density_map, fwhm=fwhm_rad)
    if np.max(smoothed_map) > 0:
        smoothed_map /= np.max(smoothed_map)
    plt.figure(figsize=(10, 6))
    hp.mollview(
        smoothed_map, 
        title=title, 
        cmap='plasma',
        min=0,
        max=1,
    )
    plt.title("ALL FFP10 DIPOLES", fontsize=18, pad=20)
    fig = plt.gcf()                
    cbar_ax = fig.get_axes()[-1] 
    cbar_ax.tick_params(labelsize=18)
    hp.graticule(45,60)