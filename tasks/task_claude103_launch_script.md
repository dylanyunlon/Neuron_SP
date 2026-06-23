# Claude-103: ags1 启动脚本完善

## 任务
完善 `launch_7b.sh` 和 `setup_ags1.sh`。

## 具体工作
1. `cat launch_7b.sh` 和 `cat setup_ags1.sh` 先读
2. 在 launch_7b.sh 中:
   - CUDA_VISIBLE_DEVICES=0,1,2,3,4 (全部5张卡)
   - NUMA 绑定: GPU0-2 → NUMA0, GPU3-4 → NUMA1
   - DeepSpeed launcher 配置 (--num_gpus=5)
   - 指向 experiments/configs/7b_ags1_desloc.yaml
3. 在 setup_ags1.sh 中:
   - pip install 依赖
   - 检查 CUDA 版本兼容性 (需要 CUDA 13.0 for Blackwell SM12.0)
   - 数据集预处理步骤

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
