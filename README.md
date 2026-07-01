# 113.58M 预备最佳方案提交包

本文件夹用于保存当前项目的 113.58M 版本预备提交内容。

## 组内成员分工与贡献表

> 提交前请把“小组成员”和“学号”替换为真实信息。三名成员均参与系统开发与算法实现，任务量基本一致。

| 小组成员 | 学号 | 分工 | 贡献占比 |
| --- | --- | --- | --- |
| 待填写 | 待填写 | 负责 RGB-T 数据处理与训练数据构建，包括 RGB/热红外图像配对、YOLO 格式数据组织、类别映射检查、训练与验证数据配置维护，并参与训练脚本调试。 | 33.3% |
| 待填写 | 待填写 | 负责 RGB-T 双分支检测模型开发，包括 YOLO11l dual-stem 结构接入、双模态特征融合模块实现、预训练权重加载适配、训练参数配置与模型收敛调试。 | 33.3% |
| 待填写 | 待填写 | 负责推理与结果融合系统开发，包括 none/hflip 测试时增强、COCO JSON 结果生成、WBF 检测框融合、提交文件压缩与线上评测结果分析。 | 33.4% |

## 推荐提交

优先提交：

```text
01_submission_json/RECOMMENDED_submit_113M_score_0p370549.json
```

网页中的“模型大小，单位 M”填写：

```text
113.58
```

该方案线上分数约：

```text
mAP = 0.370549
```

## 备用提交

如果需要备用 JSON，可以提交：

```text
01_submission_json/backup_submit_113M_recluster_score_0p3702.json
```

模型大小同样填写：

```text
113.58
```

该版本分数约为：

```text
mAP = 0.3702
```

## 文件夹内容说明

```text
01_submission_json/
```

存放网页评审系统可直接上传的 JSON 文件。

```text
02_weights/
```

存放该方案对应的主要训练权重：

```text
best_yolo11l_dualstem_i896.pt
```

```text
03_raw_predictions/
```

存放该权重在 test 集上的原始推理结果：

```text
raw_none.json
raw_hflip.json
```

最终推荐 JSON 是由这些预测经过 WBF 融合和压缩后得到的。

```text
04_code/
```

存放复现训练、推理、融合、评估需要的主要代码和配置。

```text
05_docs/
```

存放方案说明文档。

## 方案简述

该方案使用 YOLO11l dual-stem RGB-T 检测模型。RGB 图像和热红外图像分别经过早期特征提取分支，再在特征层进行融合，之后共享 YOLO 检测主干、颈部和检测头。

推理阶段使用整图推理和水平翻转推理，并通过 WBF 进行检测框融合。该方案没有使用 train+val 过拟合版本，也没有使用后续分数下降的 trainval JSON。

## 注意

不要提交以下失败或不稳定方向作为本预备方案：

```text
trainval_i896 相关 JSON
YOLO11x 降分版本
单独伪标签模型 JSON
```

如果只提交本方案，请以 `RECOMMENDED_submit_113M_score_0p370549.json` 为准。
