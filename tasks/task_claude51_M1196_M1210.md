# Claude-51: M1196-M1210 — Wiring: setup_hetero_mimo_training 接入 train_three_stage

## 任务
在 `pipeline/train_three_stage.py` 中调用 `setup_hetero_mimo_training()`。

## 具体工作
1. `cat pipeline/train_three_stage.py` 先读(603行)
2. `grep -n "def setup_hetero_mimo_training" deepspeed/runtime/hetero_mimo_training_loop.py` 找签名
3. 在 train_three_stage.py 的训练入口处调用 setup_hetero_mimo_training(), 传入 engine 和 config
4. 验证语法: `python3 -c "import ast; ast.parse(open('pipeline/train_three_stage.py').read())"`

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
