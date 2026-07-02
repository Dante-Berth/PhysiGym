# PhysiGym Deferred Video Compilation — TODO Checklist

## 🎯 Goal
Apply deferred video compilation to PhysiGym (same as PhysiCell) for +20-30% faster training

## 📋 Tasks

### Task 1: Update wrapper.py
**File:** `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/wrapper.py`

**Subtasks:**
- [ ] Line ~320: Add `self._pending_videos = []` in `__init__`
  ```python
  self._pending_videos = []  # [{out_dir, run_idx, type_mode, frame_buffer}, ...]
  ```

- [ ] Line ~260-280: Update class docstring to mention deferred compilation
  - [ ] Change: "save_data() renders all frames to PNG in a temp dir then calls ffmpeg"
  - [ ] To: "save_data() queues frames; compile_pending_videos() renders them later"

- [ ] Line ~670-703: Modify `save_data()` method
  - [ ] Find: `if self.generate_physicell_data and self._frame_buffer: self._compile_video(...)`
  - [ ] Replace with:
    ```python
    if self.generate_physicell_data and self._frame_buffer:
        self._pending_videos.append({
            "out_dir": out_dir,
            "run_idx": run_idx,
            "type_mode": finished_type_mode,
            "frame_buffer": [f.copy() for f in self._frame_buffer],
        })
    ```

- [ ] Line ~719: Modify `_compile_video()` signature
  - [ ] Change: `def _compile_video(self, out_dir: str, run_idx: int, type_mode: str = None):`
  - [ ] To: `def _compile_video(self, out_dir: str, run_idx: int, type_mode: str = None, frame_buffer: list = None):`
  - [ ] Add at start of method:
    ```python
    if frame_buffer is None:
        frame_buffer = self._frame_buffer
    ```
  - [ ] Update loop: `for i, frame in enumerate(frame_buffer):` (instead of `self._frame_buffer`)

- [ ] Line ~780 (end of class): Add new method
  ```python
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

- [ ] Verify syntax: `python -m py_compile wrapper.py`

---

### Task 2: Update run.py
**File:** `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/run.py`

**Subtasks:**
- [ ] Find the main training loop end (search for cleanup/finally block)
- [ ] Add comment block:
  ```python
  # ── Video compilation (deferred) ────────────────────────
  # The wrapper in the actor subprocess has queued pending videos
  # during training. They're not accessible here since they're in a
  # subprocess. Instead, compile them post-training with video_maker.py:
  #
  #   python video_maker.py --base-dir data/
  #
  # This processes both deferred frames/ and legacy SVG snapshots.
  # It's parallelizable and runs outside the training loop.
  ```

- [ ] If there's a `run_random_policy()` function, add same comment at its end
- [ ] Verify syntax: `python -m py_compile run.py`

---

### Task 3: Update run.sh
**File:** `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/run.sh`

**Subtasks:**
- [ ] Go to end of file
- [ ] Add:
  ```bash
  # ── 3. Compile all pending videos (deferred compilation) ──────────
  # After all training runs complete, compile the queued videos
  # This processes frames captured during test episodes.
  echo "============================================================"
  echo "  Compiling all pending videos..."
  echo "============================================================"
  python video_maker.py --base-dir data/

  echo "============================================================"
  echo "  All training and video compilation complete!"
  echo "============================================================"
  ```

- [ ] Verify syntax: `bash -n run.sh`

---

### Task 4: Add video_maker.py
**File:** `PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/video_maker.py`

**Choose ONE option:**

**Option A: Copy file (easier)**
- [ ] Copy from PhysiCell:
  ```bash
  cp /home/alex/Physi/PhysiCell/video_maker.py \
     /home/alex/Physi/PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/video_maker.py
  ```
- [ ] Update BASE_DIR in new file:
  - [ ] Find: `BASE_DIR = Path("/home/alex/Physi/PhysiCell/data")`
  - [ ] Change to: `BASE_DIR = Path("data")` or appropriate path for PhysiGym

**Option B: Create symlink (cleaner)**
- [ ] Create symlink:
  ```bash
  cd /home/alex/Physi/PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/
  ln -s /home/alex/Physi/PhysiCell/video_maker.py video_maker.py
  ```
- [ ] Update run.sh to use full path:
  ```bash
  python /home/alex/Physi/PhysiCell/video_maker.py --base-dir data/
  ```

- [ ] Verify syntax: `python -m py_compile video_maker.py`

---

### Task 5: Test
**Subtasks:**
- [ ] Run a short training test:
  ```bash
  cd /home/alex/Physi/PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/
  python run.py --seed 999 --total_timesteps 1000 [other args]
  ```

- [ ] Check for frames directories:
  ```bash
  find data/ -type d -name "frames" | head -5
  ```
  Should show deferred frames (not compiled yet)

- [ ] Compile videos:
  ```bash
  python video_maker.py --base-dir data/
  ```

- [ ] Verify video.mp4 files exist:
  ```bash
  find data/ -name "video.mp4" | head -5
  ```

- [ ] Verify frames/ directories are cleaned up:
  ```bash
  find data/ -type d -name "frames"
  ```
  Should be empty

---

## ✅ Completion Checklist

- [ ] wrapper.py modified (5 changes)
- [ ] run.py modified (1 comment block)
- [ ] run.sh modified (1 command block)
- [ ] video_maker.py added/linked
- [ ] All syntax verified
- [ ] Short test passed
- [ ] Videos compiled successfully

---

## 📊 Summary

**Changes needed:**
- wrapper.py: 5 modifications (50-100 lines)
- run.py: 1 comment block (10-15 lines)
- run.sh: 1 command block (10-15 lines)
- video_maker.py: 1 new file (copy/link)

**Total time estimate:** 15-30 minutes

**Expected result:** +20-30% faster training (same as PhysiCell)

---

## 🚀 After Completion

Once complete, you can use:
```bash
cd /home/alex/Physi/PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/
bash run.sh
# ... training runs + automatic video compilation!
```

Or manually:
```bash
python run.py [args]
python video_maker.py --base-dir data/
```

---

## 📚 Reference

See corresponding PhysiCell files for exact implementation:
- `PhysiCell/custom_modules/physigym/physigym/envs/wrapper.py`
- `PhysiCell/custom_modules/physigym/physigym/envs/run.py`
- `PhysiCell/run.sh`
- `PhysiCell/video_maker.py`
