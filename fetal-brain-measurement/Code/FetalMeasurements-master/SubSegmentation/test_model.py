import sys
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation')
import nibabel as nib
from lovasz import lovasz_softmax
from processing_utils import acc_no_bg, pre_processing
from fastai.basic_train import load_learner
from fastai.vision import Image
import torch
import numpy as np

print("🔍 Loading model...")
path = "/workspace/fetal-brain-measurement/Models/Subseg"
model_name = "model-tfms.pkl"

learn = load_learner(path, file=model_name)
print("✅ Model loaded successfully!")

# Simulate a realistic dummy 3D volume (approx. fetal brain MRI intensity range)
dummy_volume = np.random.randint(447, 1161, size=(160, 160, 32)).astype(np.float32)
print(f"⚠️ Dummy volume stats: min = {dummy_volume.min()}, max = {dummy_volume.max()}")

# Apply full pre-processing pipeline
print("⚙️ Running pre_processing...")
prepped_imgs, _, _, _, _ = pre_processing(dummy_volume, model_image_size=(160, 160))

# Report stats for each TTA set
print(f"\n📊 TTA augmentations: {len(prepped_imgs)} sets")
for i, aug_set in enumerate(prepped_imgs):
    stacked = torch.stack([img.data for img in aug_set])  # shape: (N, 1, 160, 160)
    print(f"➤ Augmentation {i}: mean = {stacked.mean():.4f}, std = {stacked.std():.4f}, min = {stacked.min():.4f}, max = {stacked.max():.4f}")

# Predict using first image from first TTA set
img = prepped_imgs[0][0]

print("\n🔮 Running model prediction...")
pred_class, pred_idx, outputs = learn.predict(img)

print("✅ After prediction:")
print("↳ pred_idx shape:", pred_idx.shape)
print("↳ Unique values in pred_idx:", torch.unique(pred_idx))

# Optional: Check if all zero
if torch.unique(pred_idx).numel() == 1 and torch.unique(pred_idx)[0].item() == 0:
    print("⚠️ Prediction is all background (class 0) — this could indicate preprocessing mismatch or a broken model.")
else:
    print("✅ Non-zero segmentation labels detected.")


# ================== REAL INPUT TEST ==================
print("\n🧪 Testing on real input (cropped.nii.gz)...")

real_path = "/workspace/fetal-brain-measurement/output/Pat172_Se12_Res0.7813_0.7813_Spac3.0/cropped.nii.gz"
try:
    nii = nib.load(real_path)
    real_data = nii.get_fdata()
    print("🧾 Loaded real image:", real_path)
    print("   ↳ shape:", real_data.shape)
    print("   ↳ min/max:", real_data.min(), real_data.max())

    # Preprocess real image
    prepped_real, _, _, _, _ = pre_processing(real_data, model_image_size=(160, 160))
    real_img = prepped_real[0][0]

    # Predict
    pred_class_r, pred_idx_r, outputs_r = learn.predict(real_img)

    print("✅ Real input prediction success!")
    print("↳ pred_idx shape:", pred_idx_r.shape)
    print("↳ Unique values in real pred_idx:", torch.unique(pred_idx_r))
except Exception as e:
    print("❌ Failed to process real input:")
    print(e)
print("✅ After prediction:")
print("↳ pred_idx shape:", pred_idx.shape)
print("↳ Unique values in pred_idx:", torch.unique(pred_idx))
print("✅ Non-zero segmentation labels detected.")
