import torch
import numpy as np
import nibabel as nib
import sys
import os
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation')

from fastai.basic_train import load_learner
from processing_utils import pre_processing
from torch.nn import functional as F

# Paths
pkl_path = "/workspace/fetal-brain-measurement/Models/Subseg/model-tfms.pkl"
pt_path  = "/workspace/fetal-brain-measurement/Models/Subseg/subseg_model.pt"
real_input_path = "/workspace/fetal-brain-measurement/output/Pat172_Se12_Res0.7813_0.7813_Spac3.0/cropped.nii.gz"

# Step 1: Load FastAI Learner
print("🔍 Loading FastAI learner...")
learn = load_learner(os.path.dirname(pkl_path), file=os.path.basename(pkl_path))
model = learn.model.cpu().eval()
print("✅ Learner loaded.")

# Step 2: Export as TorchScript (.pt)
print("💾 Converting to TorchScript...")
dummy_input = torch.rand(1, 3, 160, 160)
traced_model = torch.jit.trace(model, dummy_input)
torch.jit.save(traced_model, pt_path)
print(f"✅ Saved TorchScript model to: {pt_path}")

# Step 3: Load the .pt model
print("📦 Loading exported TorchScript model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(pt_path).to(device).eval()
print("✅ TorchScript model loaded.")

# Step 4: Dummy Input Test
dummy_volume = np.random.randint(447, 1161, size=(160, 160, 32)).astype(np.float32)
print("\n🧪 Dummy volume stats:")
print("↳ shape:", dummy_volume.shape, "| min/max:", dummy_volume.min(), dummy_volume.max())

print("⚙️ Running pre_processing...")
prepped_imgs, _, _, _, _ = pre_processing(dummy_volume, model_image_size=(160, 160))

for i, aug_set in enumerate(prepped_imgs):
    img_tensor = aug_set[0].data.unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    print(f"➤ Aug {i}: shape {img_tensor.shape}, mean={img_tensor.mean():.2f}, std={img_tensor.std():.2f}")

    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze()

    print("   ↳ pred_idx unique values:", torch.unique(pred_idx))

# Step 5: Real Input Test
try:
    print("\n🧪 Testing real input:", real_input_path)
    nii = nib.load(real_input_path)
    real_data = nii.get_fdata()
    print("   ↳ shape:", real_data.shape, "| min/max:", real_data.min(), real_data.max())

    prepped_real, _, _, _, _ = pre_processing(real_data, model_image_size=(160, 160))
    real_img_tensor = prepped_real[0][0].data.unsqueeze(0).repeat(1, 3, 1, 1).to(device)

    with torch.no_grad():
        output = model(real_img_tensor)
        pred_idx = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze()

    print("✅ Real prediction done.")
    print("   ↳ pred_idx shape:", pred_idx.shape)
    print("   ↳ unique labels:", torch.unique(pred_idx))
except Exception as e:
    print("❌ Real input failed:")
    print(e)
