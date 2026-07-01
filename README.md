# 113.58M 预备最佳方案提交包

本文件夹用于保存当前项目的 113.58M 版本预备提交内容。

## 组内成员分工与贡献表

> 提交前请把“小组成员”和“学号”替换为真实信息；贡献占比总和建议为 100%。

| 小组成员 | 学号 | 分工 | 贡献占比 |
| --- | --- | --- | --- |
| 待填写 | 待填写 | 数据整理、模型训练、结果提交 | 待填写 |
| 待填写 | 待填写 | 模型结构设计、实验对比、文档撰写 | 待填写 |
| 待填写 | 待填写 | 代码调试、推理融合、结果分析 | 待填写 |
| 待填写 | 待填写 | 实验记录、报告整理、仓库维护 | 待填写 |

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
