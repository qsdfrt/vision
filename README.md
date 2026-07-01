# 无人机视角下 RGB-T 双光车辆检测项目

本仓库为“计算机视觉综合实践：无人机视角下的双光目标检测”项目的预备最佳提交包，包含模型权重、提交 JSON、推理与融合代码、实验说明文档和组内分工说明。

项目任务是在无人机视角下，利用可见光 RGB 图像和热红外 TIR 图像进行车辆目标检测。检测类别包括：

```text
car、truck、bus、van、freight_car
```

最终评审采用 COCO bbox mAP 作为主要排名指标。当前提交包保存的是线上表现稳定的 113.58M 版本方案。

## 1. 项目结果

推荐提交文件：

```text
submission_json/RECOMMENDED_submit_113M_score_0p370549.json
```

网页评审系统中的模型大小填写：

```text
113.58
```

线上评审结果：

```text
mAP     = 0.370549
mAP_50  ≈ 0.522
mAP_75  ≈ 0.430
```

备用提交文件：

```text
submission_json/backup_submit_113M_recluster_score_0p3702.json
```

备用版本线上分数约为：

```text
mAP = 0.3702
```

## 2. 方法概述

本项目使用 YOLO11l 作为基础检测模型，并针对 RGB-T 双光数据设计了 dual-stem 双分支输入结构。

整体流程如下：

```text
RGB 图像 ── RGB stem ┐
                     ├─ 特征级融合 ─ YOLO 主干/颈部/检测头 ─ 检测结果
TIR 图像 ── TIR stem ┘
```

模型不是简单地把 RGB 和 TIR 图像当作普通三通道图像处理，而是在前端为两种模态分别建立特征提取分支：

- RGB 分支负责提取可见光图像中的纹理、颜色和外观信息；
- TIR 分支负责提取热红外图像中的热辐射和夜间/低光照信息；
- 两个分支在浅层特征阶段进行融合；
- 融合后的特征继续进入共享的 YOLO 检测网络。

推理阶段采用：

- 原图推理；
- 水平翻转推理；
- WBF（Weighted Boxes Fusion）检测框融合；
- JSON 结果压缩，满足评审网页 5MB 文件大小限制。

该方案没有采用后续实验证明效果较差的 train+val 过拟合版本，也没有采用单独伪标签模型作为主提交结果。

## 3. 仓库结构

```text
submission_package_113M_best/
├── submission_json/
│   ├── RECOMMENDED_submit_113M_score_0p370549.json
│   └── backup_submit_113M_recluster_score_0p3702.json
├── weights/
│   └── best_yolo11l_dualstem_i896.pt
├── raw_predictions/
│   ├── raw_none.json
│   └── raw_hflip.json
├── code/
│   ├── configs/
│   │   ├── rgbt_dataset.yaml
│   │   └── rgbt_dataset_cloud.yaml
│   ├── vision_project/
│   │   ├── rgbt.py
│   │   ├── coco_eval.py
│   │   ├── coco_utils.py
│   │   └── constants.py
│   ├── train_rgbt_dualstem.py
│   ├── predict_rgbt_tta.py
│   ├── wbf_predictions.py
│   ├── evaluate_json.py
│   └── requirements.txt
├── docs/
│   ├── RGBT双光目标检测实验报告.docx
│   └── 组内成员分工与贡献表.docx
├── .gitattributes
├── .gitignore
└── README.md
```

各部分作用如下：

| 路径 | 作用 |
| --- | --- |
| `submission_json/` | 存放评审网页可直接上传的 COCO JSON 检测结果 |
| `weights/` | 存放当前方案对应的最佳模型权重 |
| `raw_predictions/` | 存放原图推理与水平翻转推理得到的原始预测结果 |
| `code/configs/` | 存放训练、验证和推理时使用的数据集配置文件 |
| `code/vision_project/rgbt.py` | RGB-T 数据读取、双分支模型结构和训练器实现 |
| `code/train_rgbt_dualstem.py` | YOLO11l dual-stem RGB-T 模型训练入口 |
| `code/predict_rgbt_tta.py` | RGB-T 成对图像推理与水平翻转 TTA 脚本 |
| `code/wbf_predictions.py` | 多个预测 JSON 的 WBF 融合脚本 |
| `code/evaluate_json.py` | 本地 COCO bbox 指标评估脚本 |
| `docs/` | 项目说明文档和组内成员分工表 |

## 4. 数据说明

数据集由课程提供，需要从课程说明中的网盘链接下载。本仓库不包含原始数据集。

原始数据结构大致为：

```text
dataset/
├── train/
│   ├── rgb/
│   ├── tir/
│   └── train.json
├── val/
│   ├── rgb/
│   ├── tir/
│   └── val.json
└── test/
    ├── rgb/
    └── tir/
```

训练时需要将 COCO 标注转换为 YOLO 训练格式，并组织为如下结构：

```text
data/yolo_rgb/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    └── val/

data/rgbt_vehicle/
├── train/tir/
├── val/tir/
└── test/tir/
```

其中 `data/yolo_rgb/images/` 保存 RGB 图像，`data/rgbt_vehicle/*/tir/` 保存同名热红外图像。模型通过文件名匹配 RGB 与 TIR 图像。

如果数据路径不同，需要修改：

```text
code/configs/rgbt_dataset.yaml
code/configs/rgbt_dataset_cloud.yaml
```

中的 `path` 和 `tir_root` 字段。

## 5. 实验环境

本项目主要依赖如下环境：

```text
Python >= 3.10
PyTorch + CUDA
Ultralytics YOLO 8.4.x
pycocotools
```

本地或云端均可运行。推荐使用带 NVIDIA GPU 的环境，例如：

```text
GPU: NVIDIA A10 / RTX 系列 / A100 等
CUDA: 12.x
Python: 3.10 或 3.12
```

安装依赖：

```bash
cd code
pip install -r requirements.txt
```

如果环境中尚未安装 PyTorch，请根据自己的 CUDA 版本先安装 PyTorch。例如 CUDA 12.1 环境可参考 PyTorch 官方安装命令。

## 6. 训练方式

训练入口为：

```text
code/train_rgbt_dualstem.py
```

示例命令：

```bash
cd code

python train_rgbt_dualstem.py \
  --data configs/rgbt_dataset.yaml \
  --model yolo11l.pt \
  --epochs 40 \
  --imgsz 896 \
  --batch 4 \
  --workers 8 \
  --device 0 \
  --project runs \
  --name rgbt_dualstem_yolo11l_i896
```

关键参数说明：

| 参数 | 含义 |
| --- | --- |
| `--data` | 数据集配置文件 |
| `--model` | YOLO 预训练模型或已有权重 |
| `--epochs` | 训练轮数 |
| `--imgsz` | 输入分辨率 |
| `--batch` | 批大小 |
| `--device` | GPU 编号 |
| `--name` | 实验输出目录名称 |

当前提交包中已经包含训练完成的权重：

```text
weights/best_yolo11l_dualstem_i896.pt
```

因此如果只是复现提交结果，不需要重新训练。

## 7. 推理方式

推理脚本为：

```text
code/predict_rgbt_tta.py
```

原图推理：

```bash
cd code

python predict_rgbt_tta.py \
  --weights ../weights/best_yolo11l_dualstem_i896.pt \
  --data configs/rgbt_dataset.yaml \
  --images path/to/data/yolo_rgb/images/test \
  --output ../raw_predictions/raw_none_reproduce.json \
  --imgsz 896 \
  --batch 8 \
  --workers 8 \
  --device 0 \
  --conf 0.001 \
  --iou 0.70 \
  --max-det 300 \
  --tta none
```

水平翻转推理：

```bash
python predict_rgbt_tta.py \
  --weights ../weights/best_yolo11l_dualstem_i896.pt \
  --data configs/rgbt_dataset.yaml \
  --images path/to/data/yolo_rgb/images/test \
  --output ../raw_predictions/raw_hflip_reproduce.json \
  --imgsz 896 \
  --batch 8 \
  --workers 8 \
  --device 0 \
  --conf 0.001 \
  --iou 0.70 \
  --max-det 300 \
  --tta hflip
```

## 8. 结果融合

使用 WBF 对原图推理和水平翻转推理结果进行融合：

```bash
cd code

python wbf_predictions.py \
  --source ../raw_predictions/raw_none.json 1.0 \
  --source ../raw_predictions/raw_hflip.json 1.0 \
  --output ../submission_json/reproduce_wbf.json \
  --iou 0.70 \
  --score-threshold 0.001 \
  --output-score-threshold 0.0018 \
  --max-det 300 \
  --confidence-type average
```

融合结果需要小于评审网页限制的 5MB。当前推荐提交文件已经满足该限制：

```text
submission_json/RECOMMENDED_submit_113M_score_0p370549.json
```

## 9. 本地评估

如果需要在验证集上评估 COCO bbox 指标，可以使用：

```bash
cd code

python evaluate_json.py \
  path/to/prediction.json \
  --annotations path/to/val.json
```

评估脚本会输出：

```text
mAP
mAP_50
mAP_75
AR@10
AR@100
AR@500
```

最终排名以网页评审系统返回的 mAP 为准。

## 10. 组内成员分工与贡献表

> 提交前请把“小组成员”和“学号”替换为真实信息。三名成员均参与系统开发与算法实现，任务量基本一致。

| 小组成员 | 学号 | 分工 | 贡献占比 |
| --- | --- | --- | --- |
| 待填写 | 待填写 | 负责课程提供数据下载后的整理与训练格式转换，包括数据解压、RGB/热红外图像路径配对、COCO 标注到 YOLO 训练格式转换、类别映射检查、训练/验证/测试配置文件维护，并参与数据读取模块调试。 | 33.3% |
| 待填写 | 待填写 | 负责 RGB-T 双分支检测模型开发，包括 YOLO11l dual-stem 结构接入、双模态特征融合模块实现、预训练权重加载适配、训练参数配置与模型收敛调试。 | 33.3% |
| 待填写 | 待填写 | 负责推理与结果融合系统开发，包括 none/hflip 测试时增强、COCO JSON 结果生成、WBF 检测框融合、提交文件压缩与线上评测结果分析。 | 33.4% |

## 11. 注意事项

1. 本仓库不包含课程原始数据集，运行训练或推理前需要自行下载并整理数据路径。
2. `weights/` 中的 `.pt` 文件使用 Git LFS 管理。
3. 推荐提交文件为：

   ```text
   submission_json/RECOMMENDED_submit_113M_score_0p370549.json
   ```

4. 网页模型大小填写：

   ```text
   113.58
   ```

5. 不建议将以下实验结果作为本版本主提交：

   ```text
   trainval_i896 相关 JSON
   YOLO11x 降分版本
   单独伪标签模型 JSON
   ```

本 README 对应的是当前仓库中的 113.58M 预备最佳方案。
