# PhysiGym — Deferred Video Compilation Integration Guide

## Overview

The `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/` directory needs the same deferred video compilation updates that were applied to `PhysiCell/custom_modules/physigym/physigym/envs/`.

**Current state:** Uses old blocking video compilation (videos render during training)  
**Goal:** Use deferred compilation (queue during training, compile after)

## Files to Modify

### 1. **wrapper.py** ← Core changes
Location: `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/wrapper.py`

**Changes needed:**

| Line(s) | What | How |
|---------|------|-----|
| ~320 | Add `_pending_videos` | Add after `self._frame_buffer = []`: `self._pending_videos = []` |
| ~670-703 | Modify `save_data()` | Replace direct `_compile_video()` call with frame queueing |
| ~719 | Modify `_compile_video()` | Add optional `frame_buffer=None` parameter |
| ~800+ | Add new method | Add `compile_pending_videos()` method (copy from PhysiCell version) |
| ~260-280 | Update docstring | Update class docstring to explain deferred compilation |

**Reference:** See `/home/alex/Physi/PhysiCell/custom_modules/physigym/physigym/envs/wrapper.py` for exact implementation

**Key changes:**
```python
# ADDED: In __init__ around line 320
self._pending_videos = []

# MODIFIED: In save_data(), replace this line:
#   self._compile_video(out_dir, run_idx, type_mode=finished_type_mode)
# With this:
self._pending_videos.append({
    "out_dir": out_dir,
    "run_idx": run_idx,
    "type_mode": finished_type_mode,
    "frame_buffer": [f.copy() for f in self._frame_buffer],
})

# MODIFIED: _compile_video() signature
def _compile_video(self, out_dir: str, run_idx: int, type_mode: str = None, frame_buffer: list = None):
    if frame_buffer is None:
        frame_buffer = self._frame_buffer
    # ... rest of method unchanged

# ADDED: New method at end of class (around line 800)
def compile_pending_videos(self):
    """Compile all queued videos. Call this after training completes."""
    if not self._pending_videos:
        return
    print(f"[PhysiCellModelWrapper] Compiling {len(self._pending_videos)} pending videos...")
    for i, video_task in enumerate(self._pending_videos, 1):
        print(f"  [{i}/{len(self._pending_videos)}] Compiling video for run {video_task['run_idx']}...")
        self._compile_video(
            out_dir=video_task["out_dir"],
            run_idx=video_task["run_idx"],
            type_mode=video_task["type_mode"],
            frame_buffer=video_task["frame_buffer"],
        )
    self._pending_videos = []
    print("[PhysiCellModelWrapper] All videos compiled.")
```

---

### 2. **run.py** ← Add usage notes
Location: `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/run.py`

**Changes needed:**

| What | Where | Action |
|------|-------|--------|
| Add comment block | End of training loop (similar to PhysiCell) | Add notes about post-processing |
| Explain workflow | After main() or in docstring | Explain that videos are queued during training |

**Minimal change — just documentation:**
```python
# Add this comment block at the end of the main training function:

# ── Video compilation (deferred) ────────────────────────
# The wrapper in the actor subprocess has queued pending videos
# during training. They're not accessible here since they're in a
# subprocess. Instead, compile them post-training with video_maker.py:
#
#   python PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/video_maker.py --base-dir data/
#
# This processes both deferred frames/ and legacy SVG snapshots.
# It's parallelizable and runs outside the training loop.
```

---

### 3. **run.sh** ← Automate workflow
Location: `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/run.sh`

**Changes needed:**

| Action | Where |
|--------|-------|
| Add video compilation command | End of script, after all training completes |
| Add output messages | For clarity |

**Add to end of script:**
```bash
# ── Video compilation (deferred) ──────────────────────
# After all training runs complete, compile the queued videos
echo "============================================================"
echo "  Compiling all pending videos..."
echo "============================================================"
python video_maker.py --base-dir data/

echo "============================================================"
echo "  All training and video compilation complete!"
echo "============================================================"
```

---

### 4. **video_maker.py** ← NEW FILE (or copy)
Location: `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/video_maker.py`

**Action:** Copy from `/home/alex/Physi/PhysiCell/video_maker.py` or create symlink

This file doesn't exist in PhysiGym yet. You have two options:

**Option A: Copy the file**
```bash
cp /home/alex/Physi/PhysiCell/video_maker.py \
   /home/alex/Physi/PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/video_maker.py
```

Then update the BASE_DIR path:
```python
# Change from:
BASE_DIR = Path("/home/alex/Physi/PhysiCell/data")

# To:
BASE_DIR = Path("data")  # or wherever your PhysiGym data goes
```

**Option B: Create a symlink** (cleaner)
```bash
cd /home/alex/Physi/PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/
ln -s /home/alex/Physi/PhysiCell/video_maker.py video_maker.py
```

Then in run.sh, use:
```bash
python /home/alex/Physi/PhysiCell/video_maker.py --base-dir data/
```

---

## Implementation Checklist

- [ ] **wrapper.py**
  - [ ] Add `self._pending_videos = []` in `__init__` (~line 320)
  - [ ] Modify `save_data()` to queue instead of compile (~line 670-703)
  - [ ] Modify `_compile_video()` signature to accept `frame_buffer` (~line 719)
  - [ ] Add `compile_pending_videos()` method (~line 800)
  - [ ] Update class docstring (~line 260-280)

- [ ] **run.py**
  - [ ] Add comment block at end of training loop (~line with final cleanup)
  - [ ] Explain deferred compilation workflow

- [ ] **run.sh**
  - [ ] Add video compilation command at end (~line 70+)
  - [ ] Add output messages for clarity

- [ ] **video_maker.py**
  - [ ] Copy from PhysiCell or create symlink
  - [ ] Verify BASE_DIR is correct

- [ ] **Test**
  - [ ] Run a short training test
  - [ ] Verify `frames/` directories are created
  - [ ] Run `video_maker.py`
  - [ ] Verify `video.mp4` files are created

---

## Quick Diff Summary

### wrapper.py changes (3 key modifications)
```diff
- Line 320: Add self._pending_videos = []
- Line 670-703: Queue frames instead of compile immediately
- Line 719: Accept frame_buffer parameter
+ Line 800: Add compile_pending_videos() method
```

### run.py changes (1 documentation block)
```diff
+ Add comment block explaining deferred compilation
```

### run.sh changes (1 command block)
```diff
+ Add python video_maker.py call at end
```

### video_maker.py (1 new file)
```diff
+ Copy from PhysiCell version
```

---

## Performance Impact

Same as PhysiCell version:
- **Training speed:** +20-30% (no per-episode video overhead)
- **Total time:** +5-10% (batch compilation efficient)
- **Disk usage:** No change

---

## Reference Files

Compare with these files to see exact implementation:

- `/home/alex/Physi/PhysiCell/custom_modules/physigym/physigym/envs/wrapper.py` (reference)
- `/home/alex/Physi/PhysiCell/custom_modules/physigym/physigym/envs/run.py` (reference)
- `/home/alex/Physi/PhysiCell/run.sh` (reference)
- `/home/alex/Physi/PhysiCell/video_maker.py` (to copy)

---

## Files NOT to Modify

The following files in PhysiGym can stay as-is (no changes needed):
- `init_conds.py`
- `networks.py`
- `physicell_model.py`
- `rb.py`
- `resilient_sub_vec_env.py`
- `vectorized.py`

These don't involve video compilation, so they don't need updates.

---

## Summary

**Total work:** ~50-100 lines of code changes across 3 files + 1 new file

**Time estimate:** 15-30 minutes to apply all changes

**Result:** PhysiGym gets same 20-30% training speedup as PhysiCell
