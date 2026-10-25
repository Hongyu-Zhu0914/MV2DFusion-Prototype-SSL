# 代码地图：不用重新寻找原文件

所有既有模型、配置和根目录入口保持原相对路径。分类入口只是薄包装，不重写模型，也不自行选择训练阶段。

## 当前三条训练路径

| 路径 | 分类入口 | 原始入口 | 主要模型 |
|---|---|---|---|
| 十类原型Stage1 | `scripts/prototype/train_stage1.py` | `train_prototype_stage1_0920.py` | `projects/mmdet3d_plugin/models/dense_heads/prototype_stage1_head_0920.py`、`models/detectors/prototype_only_detector_0920.py` |
| 十类原型Stage2 | `scripts/prototype/train_stage2.py` | `train_prototype_stage2_20261025.py` | `projects/mmdet3d_plugin/models/dense_heads/prototype_stage2_adaptive_head_20261023.py`、`models/detectors/prototype_only_stage2_20261025.py` |
| 在线EMA半监督 | `scripts/ssl/train.py` | `train_online_ema_1011.py` | `projects/mmdet3d_plugin/models/detectors/mv2dfusion_online_ema_1011.py`及增强子类 |

表中缩写的`models/detectors/`同样位于`projects/mmdet3d_plugin/`下。Stage1与Stage2分别使用对应配置的`custom_imports`，不要手动替换活动头来启动这两条当前路径。

### 半监督与增强

- 固定教师基础包装：`projects/mmdet3d_plugin/models/detectors/mv2dfusion_online_ssl_0919.py`。
- EMA更新：`projects/mmdet3d_plugin/models/detectors/mv2dfusion_online_ema_1011.py`。
- 增强包装：`projects/mmdet3d_plugin/models/detectors/mv2dfusion_online_ema_aug_2026_10_14.py`及`mv2dfusion_online_ema_aug_ab_v2_2026_10_14.py`。
- 扰动实现：`projects/mmdet3d_plugin/models/utils/student_augment_2026_10_14.py`、`student_augment_ab_v2_2026_10_14.py`。
- 真/无标注数据包装：`projects/mmdet3d_plugin/datasets/online_nuscenes_dataset_0919.py`。
- 当前全局扰动配置：`projects/configs/nusc/ssl_online_r50_ema_aug_global_2026_10_19.py`；请阅读其完整`_base_`继承链，而非只看单个配置文件。

## 评估与导出

| 分类入口 | 原实现 | 用途 |
|---|---|---|
| `scripts/evaluation/evaluate_ssl.py` | `evaluate_online_ema_splits_1011.py` | 教师/学生/R50的指定子集评估 |
| `scripts/evaluation/evaluate_subset.py` | `validate_prototype_subset_0924.py` | 自定义GT子集评估 |
| `scripts/evaluation/export_stage1.py` | `export_proto_val_iter6000_0929.py` | Stage1有效Query及投影特征 |
| `scripts/evaluation/export_stage2.py` | `export_stage2_aligned_20261025.py` | Stage2候选/最终预测分离导出及匹配 |
| `scripts/visualization/plot_stage2.py` | `plot_stage2_results_20261025.py` | Stage2分析及科学绘图 |
| `scripts/visualization/plot_stage1_similarity.py` | `visualisation/plot_iter6000_positive_vs_negative_similarity_10cls.py` | 旧Stage1分布图，区间优先匹配口径 |

`scripts/analysis/`保存近期追加的独立统计/绘图脚本。这些脚本是特定实验快照的分析入口，包含对已有导出产物、路径和哈希的严格要求；不是无数据即可运行的演示。它们没有把Stage2特征用于Stage1原型相似度计算。

## 数据工具与测试

- 数据构建：根目录`select_train_scenes_6000_0916.py`、`build_labeled_6000_0919.py`、`generate_ssl_pkls.py`、`assemble_ssl_pkls_0924.py`；原始选择结果不随仓库分发，不能重跑选择器冒充恢复同一划分。
- 分类数据入口：`scripts/data/`。
- CPU单元测试：`test_ema_unit_1011.py`、`test_online_filter_0919.py`、`test_student_augment_ab_v2_2026_10_14.py`等根目录测试脚本。
- 发布结构/语法/安全基础检查：`scripts/check_release.py`。这不替代模型训练/推理验证。

## 历史代码

`projects/mmdet3d_plugin/models/dense_heads/train_versions/`、同级`backups/`以及早期配置保留作研究回溯。哪些属于历史七类Mini、未训练变体，见`docs/HISTORICAL_VERSIONS.md`。不要因为文件日期较新就自动把它当作已验证入口。

## 为什么没有把原入口全部搬进scripts？

原训练器和导出器存在根目录定位、源码哈希审计、配置继承和兄弟模块导入。大规模搬移会改变这些行为，也会让已有实验失去直接路径对应。本次只增加分类包装与导航；原文件不改名，模型数学逻辑不重构。
