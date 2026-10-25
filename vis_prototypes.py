import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_prototypes(proto_files, class_names=None):
    """
    proto_files: list of .pth file paths, each containing a dict with 'iter' and 'prototypes'
    """
    all_data = []
    iters = []
    for f in proto_files:
        data = torch.load(f)
        iters.append(data['iter'])
        proto = data['prototypes']  # [C, K, D]
        C, K, D = proto.shape
        # 展平为 [C*K, D]
        flat = proto.view(-1, D).numpy()
        all_data.append(flat)

    # 将所有时间点的原型堆叠，以便一起降维保持一致性
    all_data = np.concatenate(all_data, axis=0)  # [总点数, D]

    # PCA 降维到2维
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_data)

    # 分离回每个时间点
    num_per_time = C * K
    splits = []
    start = 0
    for i, f in enumerate(proto_files):
        end = start + num_per_time
        splits.append(all_2d[start:end])
        start = end

    # 绘制
    colors = plt.cm.tab10(np.linspace(0, 1, C))  # 每个类别一种颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # 不同标记表示不同原型，但原型数量可能多，所以可以只用一种标记，用透明度表示时间

    fig, axes = plt.subplots(1, len(proto_files), figsize=(5 * len(proto_files), 4))
    if len(proto_files) == 1:
        axes = [axes]

    for idx, (ax, iter_val, split) in enumerate(zip(axes, iters, splits)):
        ax.set_title(f'Iteration {iter_val}')
        for c in range(C):
            # 取出该类别的所有原型点
            pts = split[c * K: (c + 1) * K]
            ax.scatter(pts[:, 0], pts[:, 1], color=colors[c], marker='o',
                       label=class_names[c] if class_names else f'Class {c}', alpha=0.7)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('work_dirs/prototype_mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc/visualisation/prototype_evolution.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    # 修改为你的保存目录
    save_dir = 'work_dirs/prototype_mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc/prototypes'
    proto_files = sorted([os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith('prototypes_iter_')])
    proto_files = proto_files  # 按迭代顺序
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                   'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']  # 与你的数据集一致
    visualize_prototypes(proto_files, class_names)