# shrink_tb.py
import sys, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter  # 若沒有 torch，可改用 tensorboardX 或安裝 torch

src = sys.argv[1]  # 原 logdir
dst = sys.argv[2]  # 新 logdir
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 50

os.makedirs(dst, exist_ok=True)
ea = EventAccumulator(src, size_guidance={'scalars': 0})  # 0 = 不限
ea.Reload()

writer = SummaryWriter(log_dir=dst)
tags = ea.Tags().get('scalars', [])
print("Found scalar tags:", tags)

for tag in tags:
    events = ea.Scalars(tag)
    # 避免非單調 step（保險做法：按 step 排序）
    events_sorted = sorted(events, key=lambda e: e.step)
    for e in events_sorted[::stride]:
        writer.add_scalar(tag, e.value, e.step)

writer.flush()
writer.close()
print("Done. Try: tensorboard --logdir", dst)
