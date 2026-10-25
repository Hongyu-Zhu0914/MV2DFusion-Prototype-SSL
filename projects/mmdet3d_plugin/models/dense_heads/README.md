# dense_heads 文件说明

本目录于 2026-10-22 按用途分类。仅移动历史源码，未修改任何原有 Python 文件；活动头没有切换。

## 根目录：实际加载入口与公共依赖

- `mv2dfusion_head.py`：基础融合检测头，最近 EMA 半监督训练使用的 MV2DFusionHead。
- `MV2D_one_stage.py`：历史原型实验的固定入口，注册 PrototypeMV2DFusionHead。训练/测试前将目标版本复制到此文件。
- `one_stage_test.py`：原型测试与特征导出头，保留原测试切换路径。
- `prototype_stage1_head_0920.py`：独立注册的 PrototypeStage1Head0920，九月 Stage1 配置通过 custom_imports 使用。
- `contrast.py`：投影头、归一化和动量更新工具。
- `sinkhorn.py`、`sinkhorn_balanced_0712.py`：原型分配算法。
- `__init__.py`：默认注册入口，内容未改动。

## train_versions/：待复制的训练头模板

这里的文件保留原始相对导入，供复制到父目录的 MV2D_one_stage.py 后使用，不要直接按子模块导入，也不要给该目录添加批量注册入口。

| 文件 | 方法/用途 |
| --- | --- |
| now_one_stage_0712.py | 早期平衡 Sinkhorn 原型训练 |
| now_one_stage_0716.py | WTA、质量权重及失活原型处理 |
| now_one_stage_0718.py | Stage3 与负原型学习 |
| now_one_stage_0723.py | Stage1、平衡 Sinkhorn 与特征队列 |
| now_one_stage_0724.py | 负样本监控与权重调整 |
| now_one_stage_0725.py | 距离条件分配与自适应统计 |
| now_one_stage_0726.py | 后续梯度负样本方案 |
| now_one_stage_0726_before_cap015.py | cap 调整前版本 |
| now_one_stage_0726_cap015_untrained.py | cap015 候选历史版本 |
| now_one_stage_0819.py | 八月 Stage1 训练版本 |
| now_one_stage_0820.py | 八月 Stage2 训练版本 |

方法概述仅作导航。实际参数需结合源码、配置和对应实验记录确认；同名类注册不会按配置日期自动选择版本。

## backups/：历史活动头备份

- `MV2D_one_stage_before_test_0819.py`：整理时与 train_versions/now_one_stage_0819.py 字节一致。
- `MV2D_one_stage_before_0726_0820.py`：整理时与根目录 MV2D_one_stage.py 字节一致，实际含特征导出功能，不能仅凭文件名认定为 0726 训练实现。

重复文件仍全部保留。整理时根目录 MV2D_one_stage.py 是含导出功能的版本，与 one_stage_test.py 并非完全相同。

## 保留原有切换方式

先确认没有正在运行的训练或测试，并自行选择与实验配置、权重匹配的版本。以下是命令示例，本次整理没有执行切换。

```bash
cd /root/MV2DFusion-main/projects/mmdet3d_plugin/models/dense_heads
# 示例：选择 0819 训练头；不是默认推荐版本
cp train_versions/now_one_stage_0819.py MV2D_one_stage.py
# 测试时切换为导出头
cp one_stage_test.py MV2D_one_stage.py
```

日期训练头原先位于根目录，现在位于 train_versions/；备份头移到 backups/。历史 RECOVERY 文档和 SHA256 清单保留原文，它们记录的是历史路径与历史状态，并不代表整理后的布局。

## 备份与验证

整理前的完整目录备份及 SHA256、旧/新路径对照存放于：
/root/MV2DFusion-main/recovery_backups/dense_heads_organize_20261022_101705

移动后验证所有原有 Python 文件 SHA256 不变，并使用 ast.parse 进行语法检查（不导入模型、不生成 pyc、不运行训练）。.ipynb_checkpoints 和 __pycache__ 原样保留。
