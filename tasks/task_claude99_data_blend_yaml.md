# Claude-99: data blend 权重写入 7b_commitpack.yaml

## 任务
在 configs/7b_commitpack.yaml 中添加 data_blend 配置。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat configs/7b_commitpack.yaml — 读现有配置
3. 在 data 段添加:
   data_blend:
     commitpack: 0.70
     starcoderdata: 0.20
     commitpackft: 0.10
4. 在 run_pretrain.py 的 _apply_yaml_config 中读取 data.data_blend 并传给 data loader

## 铁律
- 只改已有文件
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
