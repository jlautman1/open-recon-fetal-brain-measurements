import sys
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation')
import nibabel as nib
from lovasz import lovasz_softmax
from processing_utils import acc_no_bg, pre_processing
from fastai.basic_train import load_learner
from fastai.vision import Image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from torch.nn import functional as F

pkl_path = "/workspace/fetal-brain-measurement/Models/Subseg/model-tfms.pkl"
pt_path = "/workspace/fetal-brain-measurement/Models/Subseg/subseg_model.pt"
#real_input_path = "/workspace/fetal-brain-measurement/Inputs/Fixed/Pat172_Se12_Res0.7813_0.7813_Spac3.0.nii.gz"
real_input_path = "/workspace/fetal-brain-measurement/output/Pat172_Se12_Res0.7813_0.7813_Spac3.0/cropped.nii.gz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- STEP 1: LOAD REAL INPUT -----------------
print("📥 Loading real image:", real_input_path)
nii = nib.load(real_input_path)
real_data = nii.get_fdata()
print("↳ shape:", real_data.shape, "| min/max:", real_data.min(), real_data.max())

print("⚙️ Preprocessing input...")
prepped_real, _, _, _, _ = pre_processing(real_data, model_image_size=(160, 160))
image_tensor = prepped_real[0][0].data.unsqueeze(0).repeat(1, 3, 1, 1).to(device)
print("🚧 Debug: image_tensor shape:", image_tensor.shape)
print("🚧 Debug: image_tensor min/max:", image_tensor.min().item(), image_tensor.max().item())
print("🚧 Debug: image_tensor mean/std:", image_tensor.mean().item(), image_tensor.std().item())


# ----------------- STEP 2: LOAD .PKL AND RUN -----------------
print("\n📦 Loading .pkl model...")
learn = load_learner(os.path.dirname(pkl_path), file=os.path.basename(pkl_path))
pkl_model = learn.model.to(device).eval()
#print("model.named_parameters()).keys(): ",dict(pkl_model.named_parameters()).keys())
#print("model.parameters: ",next(pkl_model.parameters()))

with torch.no_grad():
    pkl_output = pkl_model(image_tensor)

    print("🚧 Raw logits stats:")
    print("  shape:", pkl_output.shape)
    print("  min/max:", pkl_output.min().item(), "/", pkl_output.max().item())

    pkl_probs = F.softmax(pkl_output, dim=1)
    print("🚧 Softmax output stats:")
    print("  shape:", pkl_probs.shape)
    print("  class 0 mean prob:", pkl_probs[:, 0].mean().item())
    if pkl_probs.shape[1] > 1:
        print("  class 1 mean prob:", pkl_probs[:, 1].mean().item())
    if pkl_probs.shape[1] > 2:
        print("  class 2 mean prob:", pkl_probs[:, 2].mean().item())

    import matplotlib.pyplot as plt

    input_slice = image_tensor.cpu().numpy()[0, 0]
    class1_prob = pkl_probs[0, 1].cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(input_slice, cmap="gray")
    plt.imshow(class1_prob, alpha=0.5, cmap="jet")
    plt.title("Overlay: Input + Class 1 Probabilities")
    plt.savefig("/workspace/fetal-brain-measurement/output/class1_overlay.png")
    pkl_pred = torch.argmax(pkl_probs, dim=1).squeeze()

print("✅ .pkl model prediction done.")
print("↳ Unique values:", torch.unique(pkl_pred).cpu().numpy())

# ----------------- STEP 3: CONVERT TO .PT -----------------
print("\n💾 Converting model to TorchScript...")
traced_model = torch.jit.trace(pkl_model.cpu(), torch.rand(1, 3, 160, 160))
torch.jit.save(traced_model, pt_path)
print(f"✅ Saved as {pt_path}")

# ----------------- STEP 4: LOAD .PT AND RUN -----------------
print("📦 Loading .pt model...")
pt_model = torch.jit.load(pt_path).to(device).eval()

with torch.no_grad():
    pt_output = pt_model(image_tensor)
    pt_pred = torch.argmax(F.softmax(pt_output, dim=1), dim=1).squeeze()

print("✅ .pt model prediction done.")
print("↳ Unique values:", torch.unique(pt_pred).cpu().numpy())

# ----------------- STEP 5: COMPARE -----------------
print("\n🧪 COMPARISON")
print(f"➤ PKL pred sum: {pkl_pred.sum().item()}")
print(f"➤ PT  pred sum: {pt_pred.sum().item()}")

if torch.all(pt_pred == 0) and not torch.all(pkl_pred == 0):
    print("🚨 ERROR CONFIRMED: TorchScript model returns all background!")
elif torch.equal(pt_pred, pkl_pred):
    print("✅ SUCCESS: .pt matches .pkl output exactly.")
else:
    diff = (pt_pred != pkl_pred).sum().item()
    print(f"⚠️ WARNING: Predictions differ at {diff} pixels.")
