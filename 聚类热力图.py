import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1) 模拟 1000 个数据：250 x 4
# =========================
np.random.seed(42)
n_rows = 250
samples = ["eg_1", "eg_2", "eg_1", "eg_2"]

base = np.random.normal(0, 1, size=(n_rows, 1))
noise = np.random.normal(0, 0.6, size=(n_rows, 4))

effect = np.zeros((n_rows, 4))
up_idx = np.random.choice(n_rows, size=n_rows // 3, replace=False)
down_pool = list(set(range(n_rows)) - set(up_idx))
down_idx = np.random.choice(down_pool, size=n_rows // 3, replace=False)

effect[up_idx, 2:] += 2.0
effect[down_idx, 2:] -= 2.0

X = base + noise + effect

# 行方向 z-score
Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)

# 控制色条范围（更像示例）
vmin, vmax = -1.5, 1.5
Xz = np.clip(Xz, vmin, vmax)

# =========================
# 2) clustermap（关闭默认颜色条）
# =========================
sns.set(style="white")

g = sns.clustermap(
    Xz,
    cmap="RdBu_r",
    center=0,
    figsize=(7.5, 8.5),          # 稍微加宽/加高，底部标签更舒服
    yticklabels=False,
    xticklabels=samples,
    dendrogram_ratio=(0.25, 0.08),
    cbar_pos=None,               # 关键：关闭默认色条
    vmin=vmin,
    vmax=vmax
)

# =========================
# 3) 调整布局：给右侧和底部留足空间（解决底部标签被裁掉）
# =========================
g.fig.subplots_adjust(right=0.86, bottom=0.18, top=0.98)

# x 轴标签：旋转 + 对齐 + 往下挪(增大pad)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center")
g.ax_heatmap.tick_params(axis="x", pad=8)

# =========================
# 4) 手动把颜色条放到图外右侧
# =========================
cax = g.fig.add_axes([0.88, 0.22, 0.03, 0.6])  # [left, bottom, width, height]
cb = plt.colorbar(g.ax_heatmap.collections[0], cax=cax)
cb.set_label("Z-score", rotation=90)

# =========================
# 5) 保存/显示（pad_inches 防止 tight 裁切掉文字）
# =========================
plt.savefig("cluster_heatmap_method4_optimized.png", dpi=200,
            bbox_inches="tight", pad_inches=0.2)
plt.show()

