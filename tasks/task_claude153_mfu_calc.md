# Task C153: Fix MFU calculation

## Context
git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP

Log shows MFU=0.3462 which is impossible on PCIe-only heterogeneous cluster.
The calc divides cluster_peak by world_size then uses per-rank tokens — this
gives per-rank MFU against averaged peak, not true cluster MFU.

## Task
In desloc_engine.py, find the MFU section (search `_mfu`). Fix:
- Use this rank's own tier BF16 TFLOPS as _peak_flops_per_device
- Compute actual_flops from this rank's actual tokens processed
- For cluster MFU: all_reduce actual_flops sum, divide by sum(all_peak)

## Constraint
Do NOT open new branches.
