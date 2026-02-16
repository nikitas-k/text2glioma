## [Bug] DDP buffer sync bumps version counters, breaking multi-forward-before-backward

### 🐛 Describe the bug

`DistributedDataParallel` with `broadcast_buffers=True` (default) raises a spurious
`RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation` when the same module is forwarded **multiple
times** before `backward()`.

This is caused by `BroadcastWork::finish()` in
[`torch/csrc/distributed/c10d/comm.cpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/comm.cpp#L25-L42)
calling `copy_()` on the original buffer tensors — a dispatcher-tracked in-place
operation that bumps version counters via the `ADInplaceOrView` dispatch key.

**DataParallel already fixed the identical class of bug** in
[`torch/csrc/cuda/comm.cpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/comm.cpp#L125-L143)
— see `NOTE [ Version Counter in comm.*_coalesced ]`.  The DDP codepath in
`c10d/comm.cpp` was never given the same treatment.

### Version counter timeline (why the crash happens)

| Step | Operation | `running_mean` version |
|------|-----------|------------------------|
| 1a | DDP `_pre_forward()` → `BroadcastWork::finish()` → `copy_()` | **0 → 1** |
| 1b | cuDNN BN forward 1 — raw CUDA ptr write, no dispatcher | 1 (autograd saves **V=1**) |
| 2a | DDP `_pre_forward()` → `BroadcastWork::finish()` → `copy_()` | **1 → 2** |
| 2b | cuDNN BN forward 2 — raw CUDA ptr write, no dispatcher | 2 (autograd saves **V=2**) |
| 3  | `backward()` on output from step 1b: unpacks saved V=**1**, finds **2** | **💥 CRASH** |

The crash **only** manifests under DDP (not single-GPU) because the version bump
comes from the DDP buffer broadcast, not from the BN forward itself.

### C++ root cause

```
DDP.forward()
  → _pre_forward()
    → _check_sync_bufs_pre_fwd()
      → _sync_buffers()
        → _distributed_broadcast_coalesced()
          → dist._broadcast_coalesced()           # Python → C++
            → c10d::broadcast_coalesced()          # comm.cpp L66
              → BroadcastWork(…)                   # constructor: flatten + broadcast
              → BroadcastWork::finish()            # comm.cpp L25
                → unflatten_dense_tensors(…)
                → bucket_tensors_[i].copy_(…)      # ← dispatches through ADInplaceOrView
                                                   #   → increment_version(self)
```

The `copy_()` in `BroadcastWork::finish()` dispatches through
[`ADInplaceOrView::copy_()`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/VariableTypeManual.cpp#L386-L399),
which unconditionally calls `torch::autograd::increment_version(self)`.

### Proposed fix

Add `at::AutoDispatchBelowADInplaceOrView guard;` before the copy loop in
`BroadcastWork::finish()`.  This skips the `ADInplaceOrView` dispatch key,
preventing the version bump — exactly matching the semantics of the existing
DataParallel fix.

```cpp
void finish() {
    work_->wait();

    // NOTE [ Version Counter in DDP buffer sync ]
    //
    // Dispatch below ADInplaceOrView so that copy_() does not bump version
    // counters on the destination buffer tensors.  Without this guard,
    // DDP's pre-forward buffer broadcast increments version counters on
    // module buffers such as BatchNorm's running_mean / running_var.
    // When the same module is forwarded multiple times before backward()
    // (e.g. GAN discriminator training), the version mismatch between the
    // saved reference and the current tensor triggers a spurious
    // "modified by an inplace operation" RuntimeError in autograd.
    //
    // This mirrors the analogous fix for DataParallel in cuda/comm.cpp:
    //   See NOTE [ Version Counter in comm.*_coalesced ]
    at::AutoDispatchBelowADInplaceOrView guard;

    // Copy the output of the broadcast operation back.
    auto output_tensors = torch::utils::unflatten_dense_tensors(
        flat_tensor_.front(), bucket_tensors_);
    TORCH_INTERNAL_ASSERT(output_tensors.size() == bucket_tensors_.size());
    for (const auto i : c10::irange(output_tensors.size())) {
      if (output_tensors[i].numel() != 0) {
        bucket_tensors_[i].copy_(output_tensors[i], /*non_blocking=*/true);
      }
    }
  }
```

The only additional include needed is:

```cpp
#include <ATen/core/LegacyTypeDispatch.h>
```

### Reproducer

Save as `repro.py` and run with `torchrun --standalone --nproc_per_node=2 repro.py`:

```python
"""DDP + BatchNorm multi-forward-before-backward reproducer."""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.LeakyReLU(0.2),
        nn.Conv2d(8, 1, 3, padding=1),
    ).cuda()

    ddp = DDP(model, device_ids=[rank])
    opt = torch.optim.Adam(ddp.parameters(), lr=1e-4)

    x1 = torch.randn(2, 1, 16, 16, device="cuda")
    x2 = torch.randn(2, 1, 16, 16, device="cuda")

    # Two forwards on the same DDP-wrapped model before backward
    out1 = ddp(x1)
    out2 = ddp(x2)

    loss = out1.mean() + out2.mean()
    loss.backward()   # RuntimeError here on PyTorch >= 2.6
    opt.step()

    if rank == 0:
        print("SUCCESS — no version counter crash")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Diagnostic: `broadcast_buffers=False` eliminates the error

Changing `DDP(model, device_ids=[rank])` to
`DDP(model, device_ids=[rank], broadcast_buffers=False)` makes the crash
disappear, confirming the buffer sync as the sole trigger.

### Workarounds (for users today)

| Workaround | Cost |
|------------|------|
| `broadcast_buffers=False` | Zero — BN stats converge naturally across GPUs |
| `SyncBatchNorm.convert_sync_batchnorm()` | Changes training dynamics |
| Concatenated forward (cat inputs → single forward → chunk outputs) | 2× discriminator memory |

### Related

- `NOTE [ Version Counter in comm.*_coalesced ]` in
  [`torch/csrc/cuda/comm.cpp` L125–143](https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/comm.cpp#L125-L143)
  — the DataParallel fix for the same class of bug
- [MONAI GenerativeModels #451](https://github.com/Project-MONAI/GenerativeModels/issues/451)
  — users hit this in medical-image GAN training (repo now archived)

### Versions

- PyTorch 2.6.0+cu124  (crash confirmed)
- PyTorch 2.5.x  (crash confirmed — autograd version checking tightened in ~2.4)
- CUDA 12.4, Python 3.9+, NCCL backend
- Single-node multi-GPU (4× H200)
