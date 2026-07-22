# PhysiGym — Deferred Video Compilation Documentation Index

This directory contains documentation for integrating deferred video compilation into the PhysiGym project.

## 📚 Documentation Files

### **WHAT_TO_MODIFY.md** ← START HERE
Plain English explanation for what needs to change.
- **Best for:** Getting a quick understanding
- **Length:** ~5 minute read
- **Content:** Overview, why changes are needed, files to modify

### **TODO_CHECKLIST.md** ← IMPLEMENTATION GUIDE
Step-by-step checklist with exact tasks.
- **Best for:** Following along while implementing
- **Length:** 2-3 pages of checkboxes
- **Content:** Exact line numbers, code snippets, verification steps

### **INTEGRATION_GUIDE.md** ← TECHNICAL REFERENCE
Detailed technical documentation.
- **Best for:** Understanding the implementation details
- **Length:** ~5 pages of technical details
- **Content:** Architecture, API reference, performance analysis

---

## 🎯 Quick Overview

**Goal:** Make PhysiGym training 20-30% faster by deferring video compilation

**Current (Slow):**
```
Episode end → save_data() → [RENDER VIDEO] → Next episode (blocked 2-5 sec)
```

**After Upgrade (Fast):**
```
Episode end → save_data() → [QUEUE FRAMES] → Next episode (instant!)
[After training] → python video_maker.py → [Render all videos]
```

---

## 📋 Files to Modify

| File | Status | Changes | Time |
|------|--------|---------|------|
| `wrapper.py` | Update | 5 modifications | 10 min |
| `run.py` | Update | 1 comment block | 3 min |
| `run.sh` | Update | 1 command | 3 min |
| `video_maker.py` | Create | Copy from PhysiCell | 2 min |

**Total time:** ~20 minutes

---

## 🚀 Getting Started

### Option 1: Quick Overview (5 min)
1. Read `WHAT_TO_MODIFY.md` (Overview section)
2. Decide if you want to proceed

### Option 2: Full Implementation (20 min)
1. Read `WHAT_TO_MODIFY.md` (full document)
2. Use `TODO_CHECKLIST.md` as your guide
3. Implement each change with checkboxes
4. Test the integration

### Option 3: Deep Understanding (45 min)
1. Read `WHAT_TO_MODIFY.md`
2. Read `INTEGRATION_GUIDE.md`
3. Compare with PhysiCell reference files
4. Use `TODO_CHECKLIST.md` for implementation
5. Test thoroughly

---

## 🔗 Reference Files (in PhysiCell)

For exact implementation details, compare with:
- `/home/alex/Physi/PhysiCell/custom_modules/physigym/physigym/envs/wrapper.py`
- `/home/alex/Physi/PhysiCell/custom_modules/physigym/physigym/envs/run.py`
- `/home/alex/Physi/PhysiCell/run.sh`
- `/home/alex/Physi/PhysiCell/video_maker.py`

---

## 📁 Project Structure

```
PhysiGym/model/tumor_immune_tcells_v2/custom_modules/physigym/
├─ WHAT_TO_MODIFY.md        ← Overview & explanation
├─ TODO_CHECKLIST.md        ← Step-by-step guide
├─ INTEGRATION_GUIDE.md     ← Technical reference
│
├─ wrapper.py               ← TO UPDATE
├─ run.py                   ← TO UPDATE
├─ run.sh                   ← TO UPDATE
├─ video_maker.py           ← TO CREATE
│
└─ [other files - no changes]
```

---

## ✅ Implementation Checklist

- [ ] Read `WHAT_TO_MODIFY.md`
- [ ] Read `TODO_CHECKLIST.md`
- [ ] Update `wrapper.py` (5 changes)
- [ ] Update `run.py` (1 comment block)
- [ ] Update `run.sh` (1 command)
- [ ] Add `video_maker.py` (copy from PhysiCell)
- [ ] Test with short training run
- [ ] Verify `video.mp4` files appear

---

## 🎓 Reading Order

**For implementers:**
1. WHAT_TO_MODIFY.md → TODO_CHECKLIST.md

**For reviewers:**
1. WHAT_TO_MODIFY.md → INTEGRATION_GUIDE.md

**For learners:**
1. WHAT_TO_MODIFY.md → INTEGRATION_GUIDE.md → TODO_CHECKLIST.md

---

## 💡 Key Concept

**Move video rendering from during training to after training**

- During training: Queue frames (instant)
- After training: Compile all videos in batch (efficient)

This eliminates the per-episode blocking that slowed down training.

---

## 🆘 Quick Reference

**"Where do I start?"**
→ Read `WHAT_TO_MODIFY.md` → then use `TODO_CHECKLIST.md`

**"How do I implement X?"**
→ Check `TODO_CHECKLIST.md` for exact steps

**"What's the technical background?"**
→ Read `INTEGRATION_GUIDE.md`

**"I want to see exact code examples"**
→ Check reference files in `/home/alex/Physi/PhysiCell/`

---

## 📊 Expected Results

After implementation:
- ✅ Training runs without slowdown
- ✅ `frames/` directories created (not compiled during training)
- ✅ `python video_maker.py` compiles all videos after training
- ✅ `video.mp4` files appear in run folders
- ✅ ~20-30% faster training

---

**Ready? Start with WHAT_TO_MODIFY.md →**
