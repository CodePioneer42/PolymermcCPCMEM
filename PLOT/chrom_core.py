
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde, spearmanr
from scipy.signal import find_peaks
import MDAnalysis as mda
from MDAnalysis.analysis import align

# --- 全局常量 ---
N_BEADS_GLOBAL = 65
PRL_PAGE_WIDTH = 7.0 
PRL_COL_WIDTH = 3.375

# ==============================================================================
# 1. 样式与配置
# ==============================================================================

def set_pub_style():
    """设置符合物理顶刊(PRL)发表要求的绘图风格"""
    mpl.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 3.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })

# ==============================================================================
# 2. 基础工具函数
# ==============================================================================

def natural_keys(text):
    """自然排序键生成"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(text))]

def read_parameter_matrix(file_path, n):
    """读取模拟输出的参数矩阵 (i, j, value)"""
    matrix = np.zeros((n, n))
    data = np.loadtxt(file_path)
    # 假设输入是 1-based index，如果是0-based请修改此处
    indices = data[:, :2].astype(int) - 1
    values = data[:, 2]
    # 对称填充
    matrix[indices[:, 0], indices[:, 1]] = values
    matrix[indices[:, 1], indices[:, 0]] = values
    return matrix

def calculate_decay_profile(matrix):
    """计算 P(s) 接触概率衰减曲线"""
    n = matrix.shape[0]
    separations = np.arange(1, n)
    avgs = []
    for s in separations:
        diag = np.diagonal(matrix, offset=s)
        avgs.append(np.nanmean(diag))
    return separations, np.array(avgs)

# ==============================================================================
# 3. 数据加载函数
# ==============================================================================

def load_hic_matrix(filepath, n_beads):
    """加载实验 Hi-C 矩阵"""
    df = pd.read_csv(filepath, sep='\s+', header=None, names=['i', 'j', 'val'])
    mat = np.zeros((n_beads, n_beads))
    # 简单的 1-based 判断
    idx_shift = 1 if df['i'].min() >= 1 else 0
    i = df['i'].values - idx_shift
    j = df['j'].values - idx_shift
    mat[i, j] = df['val'].values
    mat[j, i] = df['val'].values # 对称化
    return mat

def load_sim_structure(group_path, round_num, step=10, n_beads=N_BEADS_GLOBAL):
    """
    加载模拟结构数据核心。假设路径正确，不处理异常。
    """
    round_folder_name = f"iter_{round_num}" 
    pdb_folder = os.path.join(group_path, 'sim_out', round_folder_name)
    
    # pdb_folder = os.path.join(group_path,  round_folder_name)
    print(f"pdb_folder={pdb_folder}")
    pdb_files = sorted(glob.glob(os.path.join(pdb_folder, '*.pdb')), key=natural_keys)
    print(pdb_files)
    alpha_file = os.path.join(group_path, 'alpha_log', f"{round_num}.txt")
    contact_file = os.path.join(group_path, 'contacts', f"{round_num}.txt")

    res = {
        'alpha': read_parameter_matrix(alpha_file, n_beads),
        'contact_pro': read_parameter_matrix(contact_file, n_beads),
        'dist_matrices': [],
        'polymer_coords': [],
        'polymer_volumes': [],
    }

    u = mda.Universe(pdb_files[0], pdb_files)
    poly_sel = f"id 1-{n_beads}"
    polymer = u.select_atoms(poly_sel)
    
    # 轨迹对齐
    ref = mda.Universe(pdb_files[0])
    align.AlignTraj(u, ref, select=poly_sel, in_memory=True).run()

    for ts in u.trajectory[::step]:
        all_atoms = u.atoms
        res['polymer_coords'].append(all_atoms.positions)

        coords = polymer.positions
        # res['polymer_coords'].append(coords.copy())

        # 距离矩阵
        dists = squareform(pdist(coords))
        res['dist_matrices'].append(dists)
        
        # 凸包体积
        if len(coords) >= 4:
            try:
                hull = ConvexHull(coords)
                res['polymer_volumes'].append(hull.volume)
            except:
                res['polymer_volumes'].append(0.0)
    
    return res

# ==============================================================================
# 4. 物理指标分析函数
# ==============================================================================

def compute_boundary_prob(dist_matrices, n_beads, window=6, prom_rel=0.08, width=2):
    """计算 TAD 边界概率 (Insulation Score based)"""
    counts = np.zeros(n_beads)
    valid_n = 0
    
    for dm in dist_matrices:
        score = np.full(n_beads, np.nan)
        padded = np.pad(dm, window, mode='constant', constant_values=np.nan)
        for i in range(n_beads):
            # Insulation score logic
            region = padded[i+1 : i+1+window, i+window-window : i+window]
            score[i] = np.nanmean(region)
            
        if np.all(np.isnan(score)): continue
        
        valid_mask = ~np.isnan(score)
        clean_score = score.copy()
        clean_score[~valid_mask] = np.min(score[valid_mask])
        
        peaks, _ = find_peaks(clean_score, prominence=clean_score.ptp()*prom_rel, width=width)
        valid_peaks = peaks[valid_mask[peaks]]
        
        if len(valid_peaks) > 0:
            counts[valid_peaks] += 1
        valid_n += 1
        
    return counts / max(1, valid_n)

def compute_coordination_numbers(dist_matrices, r_c, mu):
    """计算连续配位数 (Continuous Coordination Number)"""
    cn_list = []
    for dm in dist_matrices:
        contact = 0.5 * (1 + np.tanh(mu * (r_c - dm)))
        np.fill_diagonal(contact, 0)
        cn = np.sum(contact, axis=1)
        cn_list.extend(cn)
    return np.array(cn_list)

def compute_intra_drmsd(coords_list):
    """计算组内 dRMSD 分布 (采样加速)"""
    n_frames = len(coords_list)
    n_sample = min(n_frames, 500) 
    idx = np.random.choice(n_frames, n_sample, replace=False)
    selected_coords = [coords_list[i] for i in idx]
    
    dists = [pdist(c) for c in selected_coords]
    drmsds = []
    # Vectorized dRMSD could be faster but nested loop is clearer for N=200
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            diff = dists[i] - dists[j]
            rms_val = np.sqrt(np.mean(diff**2))
            drmsds.append(rms_val)
    return np.array(drmsds)

# ==============================================================================
# 5. 通用绘图函数 (完全解耦数据与样式)
# ==============================================================================

def plot_heatmaps_row(results_dict, groups, labels_dict, metric_key, 
                      hic_ref=None, cmap='seismic'):
    """
    绘制一排热图。
    metric_key: 'contact_pro' 或 'dist_matrices'
    """
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(PRL_PAGE_WIDTH, 2), sharey=True)
    
    if n_groups == 1: axes = [axes] # Handle single plot case
    
    ims = []
    for i, (ax, grp) in enumerate(zip(axes, groups)):
        if grp not in results_dict: continue
        
        data = results_dict[grp][metric_key]
        if metric_key == 'dist_matrices': # 若是矩阵列表，取平均
            data = np.mean(data, axis=0)
            
        # 仅上三角参与相关系数计算
        triu_idx = np.triu_indices(len(data), k=1)
        if hic_ref is not None:
            r, _ = spearmanr(data[triu_idx], hic_ref[triu_idx])
            title = f"{labels_dict.get(grp, grp)}, r={r:.2f}"
        else:
            title = labels_dict.get(grp, grp)
            
        # 组装显示数据：上三角(模拟) + 下三角(参考/或对称)
        plot_data = data.copy()
        if hic_ref is not None:
            plot_data = np.triu(data, 1) + np.tril(hic_ref, -1)
            np.fill_diagonal(plot_data, np.nanmax(data))

        norm = LogNorm(vmin=1e-3, vmax=1) if metric_key == 'contact_pro' else None
        im = ax.imshow(plot_data, cmap=cmap, norm=norm, origin='upper')
        ims.append(im)
        
        ax.set_title(title)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([]); ax.set_yticks([])
        # 可以在此处添加标尺逻辑 if i==0

    # Colorbar
    cbar_ax = fig.add_axes([0.91, 0.2, 0.01, 0.6])
    label_str = r'$P_{ij}$' if metric_key=='contact_pro' else 'Distance'
    fig.colorbar(ims[0], cax=cbar_ax, label=label_str)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    return fig

def plot_kde_with_inset(data_dict, means_dict, groups, colors_dict, x_label, 
                        inset_ylabel=r'$\mu$', unit_scale=1.0, x_lim=None):
    """
    通用 KDE + 插图。
    不假设组的物理顺序，仅按照 groups 列表的顺序排列插图的 x 轴。
    """
    fig, ax = plt.subplots(figsize=(PRL_PAGE_WIDTH/3, 1.8))
    
    # 插图数据容器
    ins_idx = []
    ins_vals = []
    ins_colors = []
    
    for i, grp in enumerate(groups):
        if grp not in data_dict: continue
        
        # Scaling
        vals = data_dict[grp] * unit_scale
        mean_val = means_dict[grp] # 均值在外面已经算好了，也可以这里重算
        
        c = colors_dict.get(grp, 'blue')
        sns.kdeplot(vals, ax=ax, color=c, label=None)
        
        ins_idx.append(i)
        ins_vals.append(mean_val)
        ins_colors.append(c)
        
    ax.set_xlabel(x_label)
    ax.set_ylabel('Density')
    if x_lim: ax.set_xlim(x_lim)
    ax.set_ylim(bottom=0)
    
    # 绘制右上角插图
    ax_ins = ax.inset_axes([0.62, 0.65, 0.35, 0.3])
    ax_ins.scatter(ins_idx, ins_vals, c=ins_colors, s=8, zorder=10)
    
    # 插图装饰
    ax_ins.set_xticks(ins_idx)
    ax_ins.set_xticklabels([]) # 移除具体的组名，太挤
    # ax_ins.set_xlabel('Group ID', fontsize=5) # 只有不知道k的时候才这么叫，或者干脆不写
    ax_ins.set_ylabel(inset_ylabel, fontsize=6, labelpad=1)
    ax_ins.patch.set_alpha(0.7)
    
    # 范围调整
    if len(ins_vals) > 0:
        mn, mx = min(ins_vals), max(ins_vals)
        margin = (mx - mn) * 0.2 if mx != mn else 0.1 * abs(mx)
        if margin == 0: margin = 0.1
        ax_ins.set_ylim(mn - margin, mx + margin)
    
    plt.tight_layout()
    return fig

def plot_ps_curves(results_dict, hic_mat, groups, colors_dict, labels_dict):
    """绘制 P(s) 曲线"""
    fig, ax = plt.subplots(figsize=(PRL_PAGE_WIDTH/3, 1.8))
    
    for grp in groups:
        if grp in results_dict:
            s, p = calculate_decay_profile(results_dict[grp]['contact_pro'])
            c = colors_dict.get(grp, 'blue')
            l = labels_dict.get(grp, grp)
            ax.loglog(s, p, color=c, label=l)
            
    if hic_mat is not None:
        s_exp, p_exp = calculate_decay_profile(hic_mat)
        ax.loglog(s_exp, p_exp, 'k--', label='Exp. (Hi-C)', linewidth=1)
        
    ax.set_xlabel(r'Genomic Separation $s$ (bins)')
    ax.set_ylabel(r'$P(s)$')
    # 图例配置
    ax.legend(frameon=False, ncol=1, fontsize=5, loc='lower left')
    plt.tight_layout()
    return fig