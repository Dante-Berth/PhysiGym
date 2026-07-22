# PhysiGym — What to Modify (Plain English Summary)

## The Question
> "Could you update physigym just to explain what you should modify it is the folder PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym that may imply to modify run.py, run.sh and also wrapper i guess but also to add video maker?"

**Answer:** YES, you need to modify all four files. Here's what to do:

---

## Overview of Changes

Your PhysiGym setup currently has the **old slow video system** (renders during training). You need to upgrade it to the **new fast deferred system** (queues during training, compiles after).

This involves 4 files:

| File | Status | What to Do |
|------|--------|-----------|
| **wrapper.py** | Update | Add 5 code modifications (50-100 lines) |
| **run.py** | Update | Add 1 documentation block (~15 lines) |
| **run.sh** | Update | Add 1 command block (~15 lines) |
| **video_maker.py** | Create | Copy from PhysiCell or symlink |

---

## File-by-File Guide

### 1. wrapper.py
**Location:** `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/wrapper.py`

**Why:** This file renders videos. We need to change it from "render now" to "queue for later"

**What to change (5 things):**

1. **Add queue initialization** (~line 320 in `__init__`)
   - Add one line: `self._pending_videos = []`
   - This creates a queue for video tasks

2. **Update class documentation** (~line 270)
   - Change how the video system is described
   - Old: "renders videos during training"
   - New: "queues videos, compiles after training"

3. **Modify save_data() method** (~line 700)
   - OLD: Calls `_compile_video()` immediately (blocks)
   - NEW: Adds task to `_pending_videos` queue (non-blocking)
   - This is where the magic happens — keeps training fast

4. **Modify _compile_video() method** (~line 719)
   - Add optional `frame_buffer` parameter to signature
   - This lets the method work with pre-queued frames

5. **Add compile_pending_videos() method** (~line 800)
   - NEW method that processes all queued videos
   - Called after training completes
   - This actually does the rendering work

**Copy from:** `/home/alex/Physi/PhysiCell/custom_modules/physigym/physigym/envs/wrapper.py` (our reference)

---

### 2. run.py
**Location:** `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/run.py`

**Why:** To tell users how to compile the queued videos

**What to change (1 thing):**

1. **Add comment block** at end of training loops
   - Explain that videos are queued, not compiled
   - Tell users to run `video_maker.py` after training
   - Add ~15 lines of helpful documentation

**That's it.** No code logic changes, just documentation.

---

### 3. run.sh
**Location:** `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/run.sh`

**Why:** To automate the video compilation after training

**What to change (1 thing):**

1. **Add video compilation command** at end of script
   - After all training finishes, automatically run `video_maker.py`
   - Add ~15 lines of bash code

**Before (old):**
```bash
# Training runs
python run.py ...
# Training ends, but no videos yet!
```

**After (new):**
```bash
# Training runs
python run.py ...
# Training ends
# Automatically compile all videos
python video_maker.py --base-dir data/
# Done!
```

---

### 4. video_maker.py
**Location:** `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/video_maker.py`

**Why:** This script processes queued frames and compiles them into videos

**Status:** DOESN'T EXIST in PhysiGym yet

**Solution:** Either copy or symlink from PhysiCell

**Option A: Copy (easier)**
```bash
cp /home/alex/Physi/PhysiCell/video_maker.py \
   /home/alex/Physi/PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/video_maker.py
```
Then update the `BASE_DIR` path inside the copied file.

**Option B: Symlink (cleaner)**
```bash
cd /home/alex/Physi/PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/
ln -s /home/alex/Physi/PhysiCell/video_maker.py video_maker.py
```

Both work. Symlink is better if you want one source of truth.

---

## How It Works (Simple Overview)

### OLD SYSTEM (Current PhysiGym)
```
Episode 1 starts
Episode 1 completes
↓
save_data() called
  ├─ Write CSV ✓ (fast)
  └─ Render video ✗ (2-5 seconds - BLOCKS TRAINING!)
↓
Episode 2 starts (delayed)
```

**Result:** Training is slow due to per-episode video rendering

### NEW SYSTEM (After upgrade)
```
Episode 1 starts
Episode 1 completes
↓
save_data() called
  ├─ Write CSV ✓ (fast)
  └─ Queue frames ✓ (instant)
↓
Episode 2 starts IMMEDIATELY (no blocking!)
... more episodes run fast ...
↓
Training ends
↓
python video_maker.py
  └─ Renders all videos in batch (offline, parallelizable)
↓
Done!
```

**Result:** Training is 20-30% faster!

---

## Key Insight

**The big change:** Move video rendering from **during training** to **after training**

- During training: Just queue frames (instant)
- After training: Compile all videos at once (efficient)

This is why you get the speedup.

---

## Implementation Steps

1. **Update wrapper.py**
   - 5 modifications (lines ~320, 270, 700, 719, 800)
   - Use PhysiCell version as reference
   - ~50-100 lines

2. **Update run.py**
   - 1 documentation block (end of training loops)
   - ~15 lines
   - Just explains workflow to users

3. **Update run.sh**
   - 1 command block (end of script)
   - ~15 lines
   - Calls `video_maker.py` after training

4. **Add video_maker.py**
   - Copy or symlink from PhysiCell
   - 1 file (~250 lines)

5. **Test**
   - Run training short test
   - Check `frames/` directories created
   - Run `video_maker.py`
   - Verify `video.mp4` files exist

---

## Time Estimate

- wrapper.py modifications: 10 minutes
- run.py update: 3 minutes
- run.sh update: 3 minutes
- video_maker.py copy: 2 minutes
- Testing: 10 minutes

**Total: 15-30 minutes**

---

## Files NOT to Modify

These don't need changes:
- `init_conds.py` (generates initial conditions)
- `networks.py` (neural network code)
- `physicell_model.py` (simulation)
- `rb.py` (replay buffer)
- `resilient_sub_vec_env.py` (env wrapper)
- `vectorized.py` (parallel envs)

Only the video-related files need changes.

---

## Success Criteria

After implementing, you should see:
- ✅ Training runs at normal speed
- ✅ `frames/` directories created in run folders
- ✅ Videos NOT rendered during training
- ✅ `python video_maker.py` compiles all videos after training
- ✅ `video.mp4` files appear in run folders
- ✅ ~20-30% faster training compared to before

---

## Support

If you get stuck on any file:
- Reference: `/home/alex/Physi/PhysiCell/custom_modules/physigym/physigym/envs/wrapper.py`
- Checklist: `PhysiGym/TODO_CHECKLIST.md` (detailed step-by-step)
- Guide: `PhysiGym/INTEGRATION_GUIDE.md` (technical details)

All three documents are in the PhysiGym folder.

---

## TL;DR

**Update 4 files in `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/`:**

1. **wrapper.py** — 5 code changes + queue system
2. **run.py** — 1 comment block explaining deferred compilation
3. **run.sh** — 1 command to run video_maker.py after training
4. **video_maker.py** — Copy from PhysiCell (new file)

**Result:** Same system as PhysiCell, 20-30% faster training
