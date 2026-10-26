# 根目录脚本归类（2026-10-26）

本次仅整理独立公开仓库；远程原实验目录 `/root/MV2DFusion-main`、数据、权重和结果均不移动。模型和配置 `projects/` 保持原样，不修改算法或训练设置。

| 目录 | 内容 |
|---|---|
| `scripts/prototype/` | 原型Stage1/Stage2训练，短名称推荐入口和带日期实现 |
| `scripts/ssl/` | 固定教师、EMA教师学生训练 |
| `scripts/evaluation/` | 子集评估、特征导出、匹配及导出核验 |
| `scripts/data/` | 场景选择、PKL构建/合并/检查 |
| `scripts/visualization/` | 原型、相似度和检测可视化 |
| `scripts/diagnostics/` | 预检、烟雾测试、伪标签诊断 |
| `scripts/orchestration/` | 特定历史实验的队列和批处理启动器，不是通用一键入口 |
| `scripts/analysis/` | 已有独立对比分析 |
| `tests/unit/` | EMA、筛选和输入增强CPU单元测试 |

48个原根目录Python文件全部保留，日期文件名不变；完整逐文件映射见 [RELOCATION_MAP.json](RELOCATION_MAP.json)。例如：

```text
train_prototype_stage2_20261025.py
  → scripts/prototype/train_prototype_stage2_20261025.py
train_online_ema_1011.py
  → scripts/ssl/train_online_ema_1011.py
export_stage2_aligned_20261025.py
  → scripts/evaluation/export_stage2_aligned_20261025.py
test_ema_unit_1011.py
  → tests/unit/test_ema_unit_1011.py
```

推荐命令保持 `python scripts/prototype/train_stage2.py ...` 等短入口；也可从仓库根目录直接运行归类后的日期实现。旧根目录命令在本公开仓库中不再有效。迁移前布局可在Git提交 `0fb9471498b2de86bea2fee12df500049b652aef` 查看，无需恢复旧文件到根目录。

同步修改了跨脚本导入、仓库根目录解析、子进程路径和Stage2源码审计清单。审计仍然检查当前真实源文件，但迁移后的源码SHA256自然不同，不能冒称与旧运行源码哈希一致。`SOURCE_MANIFEST.json`保留原始来源路径和变更前校验值。

验证：运行 `python tests/unit/test_repository_layout.py` 和 `python scripts/check_release.py`。CPU模型单元测试记录见 `CPU_VALIDATION.json`；这些检查不等于在新布局下完成GPU训练或真实数据推理。历史编排脚本仍依赖原实验数据、权重和状态文件，不要盲目启动。
