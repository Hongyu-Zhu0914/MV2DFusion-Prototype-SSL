# 历史版本保留规则

以下文件保留用于方法探索回溯，不代表当前默认训练设置：

- `projects/mmdet3d_plugin/models/dense_heads/train_versions/now_one_stage_07*.py`：Mini阶段不同Sinkhorn/WTA/原型约束探索。
- `now_one_stage_0820.py`及`prototype_finetune_sinkhorn_stage2_gradneg_0726_from0819_iter5000_0820.py`：Mini阶段实际自适应负损失历史，不是当前十类Stage2参数。
- `*_cap015_untrained.py`：未训练变体，不能作为正式结果来源。
- `projects/mmdet3d_plugin/models/dense_heads/backups/`：旧头备份，保留内容，不自动激活。
- `prototype_stage1_r50_labeled6000_0920.py`：历史七类配置；当前十类配置在其上覆盖`active_proto_class_ids`等字段。
- `ssl_online_r50_fixed_teacher_0919.py`：固定教师早期脚手架；当前EMA配置通过继承切换更新机制。
- `ssl_online_r50_ema_aug_target_v2_2026_10_14.py`：目标范围扰动备选；当前选定的是2026-10-19全局策略。
- `*_before_score02_*`：调整初筛门槛前的历史配置。

当前路径见README和`docs/CODE_MAP.md`。不要将历史文件名中的5000/6000/13000与本轮Stage2新增6000、累计12000混淆。

原`MV2D_one_stage.py`仍按旧项目约定保留，没有在此次发布中自动切换头。当前Stage1、Stage2与EMA路径通过自己的配置/导入选择模块，不能随意复制历史头覆盖它们。
