# Claude-118: torch.load 缺 map_location

## Bug
desloc_engine.py:1994: torch.load() 没有 map_location。
默认 load 到保存时的 device, 多卡时全部 load 到 cuda:0 → OOM。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. grep -n "torch.load(" deepspeed/runtime/desloc_engine.py
3. 所有 torch.load 加 map_location=f"cuda:{torch.cuda.current_device()}"
4. 或者 map_location="cpu" 先加载到 CPU 再手动 .to(device) (更安全)

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: torch.load with map_location to prevent cuda:0 OOM"
- git push origin main
