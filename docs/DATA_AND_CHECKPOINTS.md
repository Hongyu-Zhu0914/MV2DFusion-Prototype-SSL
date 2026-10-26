# 数据与权重：仅说明，不分发

需要自行取得nuScenes数据和相应访问许可。此仓库没有传感器数据、标注、PKL、sample token清单、预训练权重或实验检查点。

历史研究划分如下；它们是特定保存划分，不是随机重抽即可得到的标准划分：

| 集合 | 帧数 | 用途 |
|---|---:|---|
| 有标注训练 | 6000 | 从官方train按完整scene选择150个scene |
| 无标注训练 | 6008 | 官方test，仅传感器输入，无可用人工GT |
| 自定义验证 | 3014 | 开发、诊断、阈值选择 |
| 内部有GT测试 | 3005 | 最终内部评估；与自定义val共同组成保留的官方val |

选择时考虑类别/稀有类别、实例、log和地域覆盖，不是旧13008帧方案或随机20%抽样。相同实验恢复需要保存的精确划分PKL，而不是重新运行优化器猜测名单。

## 预期路径

归类后的脚本从自身位置解析仓库根目录，不再依赖原根目录入口。历史配置和批处理仍包含数据根目录`/root/autodl-tmp/data/nuscenes`、结果目录及预检状态要求。在不同机器上使用时，需要审阅并显式调整配置/路径；不要修改含来源哈希保护的启动器来绕过检查。

```text
weights/r50.pth
/root/autodl-tmp/data/nuscenes/ssl_splits/
  nuscenes_ssl_train_labeled_150scenes_6000_0919.pkl
  nuscenes_ssl_train_unlabeled_official_test.pkl
  nuscenes_ssl_val_split.pkl
  nuscenes_ssl_test_split.pkl
work_dirs/prototype_stage1_r50_labeled6000_10cls_0923/checkpoints/iter_6000.pth
```

Stage1使用R50基线；Stage2需要完整十类Stage1检查点；在线SSL从R50初始化，并使用包含教师/学生状态的SSL检查点续训。不要把这些类型的权重互相替换，也不要只恢复原型矩阵而忽略投影和归一化层。

PKL引用的当前帧图像/点云、历史sweeps、元数据和maps必须同时可访问。没有数据的新克隆实例可以做源码检查，但不能据此宣称数据管线、模型推理或mAP/NDS已复测。
