# MV2DFusion Prototype Learning & Online Semi-Supervised Detection

基于 [MV2DFusion](https://github.com/wangzt-halo/MV2DFusion) 的研究代码：类别原型学习，以及图像–点云多模态在线教师–学生半监督三维检测。

**这是研究扩展仓库，不是MV2DFusion官方仓库。** 原始检测框架、FSDv2和基础模型属于其原作者；本仓库保留上游来源和提交历史。原上游README保存于 [docs/upstream/README.md](docs/upstream/README.md)。许可现状见 [UPSTREAM_AND_LICENSE.md](UPSTREAM_AND_LICENSE.md)。

## 当前研究状态

- 已实现并运行：基于R50检测器的十类原型Stage1，以及仅用有标注数据的自适应负损失Stage2；原型训练时检测器冻结。
- 已实现并运行：在线EMA教师–学生框架、学生全局输入扰动、按预测置信度筛选伪标签。
- **尚未接入和验证：使用已训练原型对在线教师候选做相似度精筛。** 两条分支目前独立，不能把原型Stage2当成半监督训练，也不能把旧原型直接套到新检测器的特征上。
- 仓库仅含代码、配置、使用说明和代码验证报告；不包含nuScenes数据、PKL划分文件、权重、特征NPY或私人工作记录。

详细边界见 [研究方法与状态](docs/METHOD_STATUS.md)。本仓库不把计划写成已完成结果，也不将内部子集指标表述为官方测试成绩。

## 从哪里开始

| 任务 | 当前配置 | 推荐入口 |
|---|---|---|
| 十类原型Stage1，6000步 | `projects/configs/nusc/prototype_stage1_r50_labeled6000_10cls_0923.py` | `scripts/prototype/train_stage1.py` |
| 原型Stage2，新增6000步 | `projects/configs/nusc/prototype_stage2_labeled6000_10cls_20261025.py` | `scripts/prototype/train_stage2.py` |
| EMA半监督＋学生全局扰动 | `projects/configs/nusc/ssl_online_r50_ema_aug_global_2026_10_19.py` | `scripts/ssl/train.py` |
| 教师/学生或R50子集评估 | 选择与权重对应的SSL配置 | `scripts/evaluation/evaluate_ssl.py` |
| 原型Stage2评估＋特征导出 | 选择与权重对应的原型配置 | `scripts/evaluation/export_stage2.py` |
| 原型相似度分布 | 显式指定特征、原型、GT PKL | `scripts/visualization/plot_stage1_similarity.py` |

**为保证旧实验可定位与源码审计不失效，没有移动或重命名原训练脚本、模型和配置。** `scripts/`是按任务分类的薄包装，最终仍运行原入口；完整路径对应关系见 [代码地图](docs/CODE_MAP.md)。

```text
projects/           模型、数据管线、nuScenes/Argoverse配置
tools/              上游通用训练、测试和数据转换工具
scripts/            分类入口：prototype / ssl / evaluation / data / visualization
visualisation/      原有科学绘图实现
docs/               方法状态、环境、数据、历史版本、发布验证
train_*.py          原训练入口，保留路径与原始行为
export_*.py         原特征导出入口，保留路径与原始行为
```

## 环境与准备

历史运行环境是Linux、Python 3.8.10、PyTorch 1.10.0、MMCV-full 1.6.1、MMDetection 2.25.1和MMDetection3D 1.0.0rc4。上游README的PyTorch版本与实际研究环境不同；不要混装或直接升级。参见 [环境说明](docs/ENVIRONMENT.md) 和 [依赖参考](requirements/observed-runtime.txt)。

训练需要另行安装CUDA相关扩展、下载授权数据和权重。**仓库代码可下载不代表在无GPU、无数据的新环境中已经能直接训练。** 数据和权重的预期位置见 [DATA_AND_CHECKPOINTS.md](docs/DATA_AND_CHECKPOINTS.md)。

## 运行方式

以下命令从仓库根目录执行。首先可以不导入模型，检查包装入口将要运行的原脚本：

```bash
python scripts/prototype/train_stage2.py --show-command
python scripts/ssl/train.py --show-command
python scripts/check_release.py
```

准备好环境、数据和初始权重后，使用明确配置；下列是运行示例，不是安装或GPU验证已完成的声明：

```bash
# Stage1必须显式选十类配置；旧入口的默认配置是历史版本。
python scripts/prototype/train_stage1.py \
  --config projects/configs/nusc/prototype_stage1_r50_labeled6000_10cls_0923.py

# Stage2加载完整Stage1权重，而不是仅载入原型矩阵。
python scripts/prototype/train_stage2.py \
  --config projects/configs/nusc/prototype_stage2_labeled6000_10cls_20261025.py \
  --launch-formal

# 在线EMA半监督；初始权重为R50，不是原型Stage2权重。
python scripts/ssl/train.py \
  --config projects/configs/nusc/ssl_online_r50_ema_aug_global_2026_10_19.py \
  --work-dir work_dirs/ssl_online_new_run
```

先在独立输出目录做少量预检，不要覆盖旧实验。历史启动器含本地绝对路径、校验哈希或既有预检记录要求；不要把它们当作通用的一键安装程序。参见 [代码地图](docs/CODE_MAP.md)。

## 评估与可视化口径

- 自定义验证集3014帧、内部有GT测试集3005帧，均不同于官方无GT test；不能在官方无GT test本地计算真实mAP/NDS。
- 对比时统一token范围、seed、输入sweep处理和时序策略；不要混用逐帧重置与场景连续记忆的结果。
- 原型正负分析使用独立于官方AP的分类别距离/方向匹配；明确负样本不等于全部未匹配框，红蓝数量比不是官方Precision。
- `0.2<score<0.8`既有“先筛区间再匹配”旧实现，也有“整体先匹配再筛区间”新实现。统计比较必须标明并统一口径。
- 投影特征、归一化和原型必须来自同一个合法空间；掩码、解码/topk/NMS映射和传播Query口径必须一致。

参见 [评估说明](docs/EVALUATION.md)。历史Mini版本、七类配置和未训练变体见 [HISTORICAL_VERSIONS.md](docs/HISTORICAL_VERSIONS.md)。

## 引用与来源

请引用并尊重上游工作：

```bibtex
@article{wang2024mv2dfusion,
  title={MV2DFusion: Leveraging Modality-Specific Object Semantics for Multi-Modal 3D Detection},
  author={Wang, Zitian and Huang, Zehao and Gao, Yulu and Wang, Naiyan and Liu, Si},
  journal={arXiv preprint arXiv:2408.05945},
  year={2024}
}
```

本研究的教师–学生伪标签质量分析参考了[3DIoUMatch](https://arxiv.org/abs/2012.04355)。其预测IoU筛选与本项目计划中的原型语义相似度筛选并不相同。
