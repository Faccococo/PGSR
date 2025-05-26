#!/bin/bash

# 基于 .vscode/launch.json 文件生成的命令行命令
# 默认数据集名称为 "floor"，您可以通过传递参数来修改

DATASET_BASE=${1:-"floor"}
WORKSPACE_FOLDER=$(pwd)

echo "使用数据集: $DATASET_BASE"
echo "工作目录: $WORKSPACE_FOLDER"
echo ""

# run_train 配置对应的命令
echo "=== 训练命令 (run_train) ==="
echo "CUDA_VISIBLE_DEVICES=0 python train.py \\"
echo "  -s $WORKSPACE_FOLDER/datasets/$DATASET_BASE \\"
echo "  -m outputs/$DATASET_BASE \\"
echo "  -d $WORKSPACE_FOLDER/datasets/$DATASET_BASE/depths_est \\"
echo "  --debug_tensor $WORKSPACE_FOLDER/debug_tensor/ \\"
echo "  --dn_l1_weight_init 2.0 \\"
echo "  --resolution 1 \\"
echo "  --data_device cpu \\"
echo "  --opt_pose"

# 实际执行训练命令
echo "=== 执行训练命令 ==="
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s "$WORKSPACE_FOLDER/datasets/$DATASET_BASE" \
  -m "outputs/$DATASET_BASE" \
  -d "$WORKSPACE_FOLDER/datasets/$DATASET_BASE/depths_est" \
  --debug_tensor "$WORKSPACE_FOLDER/debug_tensor/" \
  --dn_l1_weight_init 2.0 \
  --resolution 1 \
  --opt_pose
  # --data_device cpu \
