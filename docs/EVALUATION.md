# 评估与原型分析不是同一种匹配

## 官方检测指标

nuScenes AP按同类别、置信度降序的一对一XY中心距离匹配，分别使用0.5/1/2/4m阈值。每类AP对四个阈值平均，mAP对十类等权平均。它不是固定score门槛的Precision，也不使用原型分析里的45度方向约束。

NDS由50%mAP与五项各10%的TP质量分数组成，五项为平移、尺寸、朝向、速度和属性误差。总分增加可能掩盖某些类别下降，或固定置信度门槛下误检增加。

自定义子集评估仍须应用nuScenes官方类别距离、无点GT、自行车架等过滤。必须核对预测/GT token集合，不能给无GT官方test计算真实指标。

参考：<https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/detection>

## 原型分析口径

类别顺序：car、truck、construction_vehicle、bus、trailer、barrier、motorcycle、bicycle、pedestrian、traffic_cone。

正样本：score>0.2、同类别、按分数降序一对一贪心；BEV距离严格小于[1,1.2,1.5,1.5,1.5,0.5,0.5,0.4,0.35,0.25]m；类别[0,1,2,3,4,6,7]朝向误差≤45度。

明确负样本：未匹配且距任意十类GT的最近BEV距离>2m。重复框/近GT模糊框不自动划为明确负样本，因此正负数量比不能直接当作官方Precision。

相似度：在当前模型自己的`proto_proj`与`feat_norm`空间，对预测类别的10个原型取最大余弦。正常和传播Query采用统一有效性/NMS掩码；pre-topk有效候选与最终检测要分开保存。

0.2–0.8区间分析存在两种不同口径：

1. 历史0929图：先筛选区间，再在区间内匹配。
2. 新全局匹配图：先用全部score>0.2候选匹配，再对正负双方筛选0.2<score<0.8。

两者正样本可能不同。新柱形图使用`Samples/bin`总数量；旧柱形图使用每帧数量；density图则对正负两组各自归一化，纵轴不能混用。

## 可比性要求

统一数据划分、token、seed、历史sweeps、过滤、解码/NMS、阈值以及时序策略。`scene-stream`与`reset-each-frame`不是相同推理协议。固定导出指标也不保证不同GPU的逐比特一致；微小变化应报告重复实验方差，不能过度归因。
