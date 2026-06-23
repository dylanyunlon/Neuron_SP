# Claude-61: M1346-M1360 — Wiring: DesLocEngine.__init__ 接入 build_neuron_sp_config

## 任务
在 desloc_engine.py 的 DesLocEngine.__init__() 中正确调用 build_neuron_sp_config()。

## 具体工作
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. cat deepspeed/runtime/desloc_engine.py — 通读全文
3. cat deepspeed/runtime/hetero_gdn_selective_recompute.py | head -100 — 读 build_neuron_sp_config 签名
4. 在 DesLocEngine.__init__() 中找到 self.recompute_config 相关行,确保调用 build_neuron_sp_config() 并把返回值存到 self.recompute_config
5. 确保 activation checkpointing 用到这个 config

## 铁律
- 只改 deepspeed/runtime/desloc_engine.py
- 作者: dylanyunlon <dogechat@163.com>
- git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
- git commit --signoff
- git push origin main
