# Claude-107: fix primary_device → local device in train loop data transfer

## Bug
desloc_engine.py line 1607-1608: input_ids.to(self.primary_device) 把所有rank的数据发到同一个GPU。
line 1999: checkpoint恢复时 tensor也发到 primary_device。
多卡 torchrun 下每个rank应该用自己的 local device。

## 修复
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat -n deepspeed/runtime/desloc_engine.py | grep -n "primary_device" | head -20
3. 在 train() 方法开头获取 local device: _local_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
4. 把 line 1607-1608 的 self.primary_device 替换为 _local_dev
5. 把 line 1999 的 self.primary_device 替换为 _local_dev
6. 保留 self.primary_device 在其他非 tensor alloc 的地方不动

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff -m "fix: train loop data transfer uses local device not primary_device"
- git push origin main
