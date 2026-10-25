# 环境说明

本项目依赖旧版OpenMMLab与CUDA扩展，建议使用隔离环境。研究实例实际观察到的版本记录于`requirements/observed-runtime.txt`；原始上游README列出的PyTorch版本是1.11.0，实际研究实例为1.10.0，两者不可混为已经验证过的同一环境。

本次发布只在新克隆实例上执行可行的CPU/源码检查，不下载数据、不安装GPU、不启动训练，也不通过升级包修复历史环境。版本清单是环境取证，不是已验证的跨平台锁文件或一条命令安装保证。

CUDA版PyTorch、spconv、flash-attn、torch-scatter和TorchEx必须按目标CUDA/编译器安装。请先阅读上游安装说明及各依赖官方文档，不要在Python 3.11或新MMDetection3D API下直接运行这些旧配置。

源码已有部分本地绝对路径与历史哈希保护。分类包装保留原入口行为，不能自动完成数据、路径或检查点恢复。
