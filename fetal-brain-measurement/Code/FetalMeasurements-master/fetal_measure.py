from os.path import join
import numpy as np
import nibabel as nib
import torch
import builtins
import sys
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation')
import gzip
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from lovasz import lovasz_softmax
from processing_utils import acc_no_bg

from fastai.basic_train import load_learner

#learn = load_learner(model_path, model_name)
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
import textwrap
import pandas as pd
import fetal_seg
from SubSegmentation.lovasz import *
from SubSegmentation.processing_utils import *
from SubSegmentation.brain_segmentation_model import BrainSegmentationModel
import slice_select
from scipy import ndimage

import CBD_BBD
import msl

import re
from fetal_normative import normative_report_all


# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):  # catches np.bool_, np.int_, etc.
            return obj.item()
        return super().default(obj)

#FD_RE = re.compile(
#            "Pat(?P<patient_id>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+).nii")
FD_RE = re.compile(
    r"Pat(?P<patient_id>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+)\.nii\.gz")


class FetalMeasure(object):
    def __init__(self, basedir="fetal-brain-measurement/Models",
                 braigseg_roi_model="22-ROI",
                 braigseg_seg_model="24-seg",
                 subseg_model='model-tfms.pkl',
                 sliceselect_bbd_model="23_model_bbd",
                 sliceselect_tcd_model="24_model_tcd",
                 normative_csv='fetal-brain-measurement/Code/FetalMeasurements-master/Normative.csv'):
        self.fetal_seg = fetal_seg.FetalSegmentation(
            os.path.normpath(os.path.join(basedir, "Brainseg", "22-ROI")), None,
            os.path.normpath(os.path.join(basedir, "Brainseg", "24-seg")), None)
        print("model1 path: ", self.fetal_seg._model_path)
        print("model2 path: ", self.fetal_seg._model2_path)
        self.norm_df = pd.read_csv(normative_csv)
        #self.subseg_learner = load_learner(os.path.join(basedir, "Subseg"), subseg_model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.subseg = BrainSegmentationModel(
        #    (160, 160),
        #    r"\\fmri-df4\projects\Mammilary_Bodies\BrainBiometry-IJCARS-Netanell\Models\Subseg",
        #    subseg_model,
        #    device=device
        #)
        model_dir = "/workspace/fetal-brain-measurement/Models/Subseg"
        print("subseg model before init: ", subseg_model)
        self.subseg = BrainSegmentationModel((160,160), model_dir, subseg_model, device=device)
        #this is the line that if documented it prodoces 6 output files
        
        
        #self.subseg = BrainSegmentationModel((160,160), os.path.join(basedir, "Subseg"), subseg_model, device=device)
        # self.sl_bbd = slice_select.SliceSelect(
        #     model_file="/home/netanell/work/FetalMeasurements/Models/Sliceselect/19_model_bbd",
        #     basemodel='ResNet50', cuda_id=0)
        # #model_file = "/home/netanell/work/research/slice_select/models/6/epoch0029_02-02_1249_choose_acc0.9678.statedict.pkl")
        # self.sl_tcd = slice_select.SliceSelect(
        #     model_file="/home/netanell/work/research/slice_select/models/7/epoch0029_02-02_1430_choose_acc0.9672.statedict.pkl", cuda_id=0)

        self.sl_bbd = slice_select.SliceSelect(
            model_file= os.path.normpath(os.path.join(basedir, "Sliceselect", "22_model_bbd")),
            basemodel='ResNet50', cuda_id=0)
        #model_file = "/home/netanell/work/research/slice_select/models/6/epoch0029_02-02_1249_choose_acc0.9678.statedict.pkl")
        self.sl_tcd = slice_select.SliceSelect(
            model_file= os.path.normpath(os.path.join(basedir, "Sliceselect", "25_model_tcd")),
            basemodel='ResNet50', cuda_id=0)
        print("Finished to load **************************")

    def _predict_nifti_subseg(self, img_data, filename):
        IMAGE_SIZE = (160, 160)
        images, min_ax, zeros, x_ax, y_ax = pre_processing(img_data, IMAGE_SIZE)
        segmentations_result = []
        zoomratio = min_ax/IMAGE_SIZE[0]
        for image in images:
            pred_class, pred_idx, outputs = self.subseg_learner.predict(image)
            segmentations_result.append(ndimage.zoom(pred_idx.data.squeeze(), zoomratio, order=0))



        seg_img = post_processing(segmentations_result, min_ax, zeros, x_ax, y_ax, filename)

    def execute(self, in_img_file, out_dir):
        ############################################################### model checking:##############################################################
        # print("checking the model: ")
        # p = "/workspace/fetal-brain-measurement/Models/Subseg/model-tfms.pkl"
        # print("Exists:",     os.path.exists(p))
        # print("File size:",  os.path.getsize(p), "bytes")

        # device = torch.device("cpu")
        # model = BrainSegmentationModel(
        # input_size=(160,160),
        # model_path="/workspace/fetal-brain-measurement/Models/Subseg",
        # model_name="model-tfms",
        # device=device
        # )
        # # If you get here without error, the .pkl at least parsed.
        # print(model.model)    # 
        # print("model printed, finished model checking, let's goooooQQQQQQQQQQQQQQQQQQQQQQQQQQ")
        # net = model.model.model  
        # dummy = torch.rand(1,3,160,160).to(device)
        # pred = net(dummy)
        # print("Output shape:", pred.shape)   #
        
        ############################################################## Not model checking any more########################################################################
        print("inside fetal measure execute printing input and out_dir: \n", in_img_file ,"\n",out_dir)
        # Full pipeline
        # in_img_file = "/media/df3-dafna/Netanell/DemoCode/Pat02_Se06_Res0.7422_0.7422_Spac5.nii"
        # out_dir = "/media/df3-dafna/Netanell/DemoCode/Outputs"

        elem_fname = os.path.basename(in_img_file) #takes the file name out of the path
        #pat_id, ser_num, res_x, res_y, res_z = [float(x) for x in FD_RE.findall(elem_fname)[0]]
        #match = FD_RE.search(os.path.basename(os.path.dirname(elem_fname)))
        #match = FD_RE.search(os.path.basename(os.path.dirname(in_img_file)))
        match = FD_RE.search(os.path.basename(in_img_file))

        if not match:
            raise ValueError(f"Filename format not recognized: {elem_fname}")
        pat_id, ser_num, res_x, res_y, res_z = [float(x) for x in match.groups()]

        metadata = {}
        metadata["InFile"] = in_img_file
        metadata["OutDir"] = out_dir
        metadata["Resolution"] = (res_x, res_y, res_z)
        metadata["SubjectID"] = pat_id
        metadata["Series"] = ser_num
		
        print("the metada is: ", metadata)
		
        # Prep image
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.basename(in_img_file) # the same as elem_fname
        elem_nii = nib.load(in_img_file) # creates a nifti file out of the file path
        # print("elem_nii is : ", elem_nii)
        elem_nii_arr = elem_nii.get_fdata() 
        # print("elem_nii_arr is : ", elem_nii_arr)
        # print("Volume stats:")
        # print("  shape:", elem_nii_arr.shape)
        # print("  min:", np.min(elem_nii_arr))
        # print("  max:", np.max(elem_nii_arr))
        # print("  unique:", np.unique(elem_nii_arr))
        input_axes = np.argsort(-np.array(elem_nii_arr.shape))
        #print("input axes: ", input_axes)
        nib_out = nib.Nifti1Image(elem_nii_arr.transpose(input_axes), affine=np.eye(4))
        #print("nib_out: ", nib_out)
        reorint_image_niifile =  os.path.normpath(os.path.join(out_dir, fn))
        #nib.save(nib_out, reorint_image_niifile)
        if reorint_image_niifile.endswith(".gz"):
            with gzip.open(reorint_image_niifile, 'wb') as f:
                f.write(nib_out.to_bytes())
        else:
            nib.save(nib_out, reorint_image_niifile)
        print("reorint_image_niifile is : ", reorint_image_niifile)
        #Debugging
        print("✅ Starting masking the brain...")
        self.fetal_seg.predict(reorint_image_niifile, out_dir)

        # ⚠️ Fix: Rename prediction_all.nii → prediction.nii.gz if needed
        pred_all = os.path.join(out_dir, "prediction_all.nii")
        pred_fix = os.path.join(out_dir, "prediction.nii.gz")
        if os.path.exists(pred_all) and not os.path.exists(pred_fix):
            os.rename(pred_all, pred_fix)
            print("🔁 Renamed prediction_all.nii → prediction.nii.gz")

        print("✅ Finished masking the brain")
        #return #jjjjjjjjjjjjjjj

        # First stage - Segmentation
        #self.fetal_seg.predict(reorint_image_niifile, out_dir)
        seg_file =  os.path.normpath(os.path.join(out_dir, "prediction.nii.gz"))
        # roi = nib.load(seg_file).get_fdata()
        # assert roi.sum() > 0, "ERROR: stage1 ROI mask is empty!"
        # print("🧪 Checking prediction.nii.gz contents...")
        # print("   ➤ prediction shape:", roi.shape)
        # print("   ➤ prediction unique values:", np.unique(roi))
        # print("   ➤ prediction max value:", np.max(roi))
        # print("   ➤ prediction nonzero voxel count:", np.count_nonzero(roi))
        #if out_dir.endswith(".nii.gz"):
        #    correct_out_dir = os.path.dirname(out_dir)
        #else:
        #    correct_out_dir = out_dir
        #subseg_file = os.path.join(correct_out_dir, "subseg.nii.gz")

        
        #subseg_file = os.path.normpath(out_dir)
        subseg_file =  os.path.normpath(os.path.join(out_dir, "subseg.nii.gz"))
        print("subseg_file: ", subseg_file)
        #print("Msl and after cause problems for some inputs - commented out for tests")
        #return metadata
        # Second stage - Slice select
        print("✅ Starting SL_TCD slice selection...")

        sl_tcd_result = self.sl_tcd.execute(img_file=reorint_image_niifile,
                                            seg_file=seg_file,visualize= True)
        print("sl_tcd_result: ", sl_tcd_result)
        print("✅ Finished SL_TCD slice selection")
        sl_tcd_slice = int(sl_tcd_result["prediction"].values[0])
        print("sl_tcd_result[prediction].values[0]", sl_tcd_result["prediction"].values[0])
        metadata["TCD_selection"] = sl_tcd_slice
        metadata["TCD_selectionValid"] = sl_tcd_result["isValid"].values[0]
        metadata["TCD_result"] = sl_tcd_result.prob_vec.tolist()
        print("slice_select_tcd: ", sl_tcd_slice)
        sl_bbd_result = self.sl_bbd.execute(img_file=reorint_image_niifile,
                                            seg_file=seg_file, )
        sl_bbd_slice = int(sl_bbd_result["prediction"].values[0])
        metadata["BBD_selection"] = sl_bbd_slice
        metadata["BBD_result"] = sl_bbd_result.prob_vec.tolist()
        metadata["BBD_selectionValid"] = sl_bbd_result["isValid"].values[0]

        data_cropped, fullseg = self.sl_tcd.get_cropped_elem(img_file=reorint_image_niifile,
                                                             seg_file=seg_file, )
        print("✅ fullseg unique values:", np.unique(fullseg))
        print("→ cropped data shape:",     data_cropped.shape)
        print("→ cropped data min/max:",   data_cropped.min(), data_cropped.max())
        print("→ cropped data nonzero voxels:", np.count_nonzero(data_cropped))
        print("→ fullseg (ROI) sum:",      fullseg.sum(), "out of", fullseg.size)
        #print("LOOOOOOOOOOOOOOOOOK3 data_cropped seems good? ", data_cropped )
        #print("LOOOOOOOOOOOOOOOOOK3 what about fulseg??  ", fullseg )
        
        data_cropped = data_cropped.transpose([1, 2, 0])
        fullseg = fullseg.transpose([1, 2, 0])
        #print("🧪 DEBUG: real subseg input stats")
        print("↳ shape:", data_cropped.shape)
        print("↳ dtype:", data_cropped.dtype)
        print("↳ min/max:", data_cropped.min(), data_cropped.max())
        print("↳ mean/std:", np.mean(data_cropped), np.std(data_cropped))
        print("↳ unique values (if small set):", np.unique(data_cropped)[:10])
        print("✅ post-transpose fullseg unique values:", np.unique(fullseg))
        print("✅ post-transpose fullseg dtype:", fullseg.dtype)
        #print("LOOOOOOOOOOOOOOOOOK4!! data_cropped seems good after? ", data_cropped )
        #print("LOOOOOOOOOOOOOOOOOK4!! what about fulseg after??  ", fullseg )
        nii_data_cropped = nib.Nifti1Image(data_cropped, affine=np.eye(4))
        nib.save(nii_data_cropped,  os.path.normpath(os.path.join(out_dir, "cropped.nii.gz")))

        nii_seg_cropped = nib.Nifti1Image(fullseg.astype(float), affine=np.eye(4))
        print("✅ seg.cropped.nii.gz voxel sum (after cast to float):", fullseg.astype(float).sum())
        nib.save(nii_seg_cropped,  os.path.normpath(os.path.join(out_dir, "seg.cropped.nii.gz")))
        
        # Third stage - Sub segmentaion

        #self._predict_nifti_subseg(data_cropped.copy(), subseg_file)
        print("📊 Cropped input range BEFORE subseg:", data_cropped.min(), data_cropped.max())
        self.subseg.predict_nifti(data_cropped.copy().astype(np.float32), subseg_file, tta=True)
        print("seeing the subseg_file: ", subseg_file)
        subseg = nib.load(subseg_file).get_fdata()
        print("✅ subseg shape:", subseg.shape)
        print("✅ subseg unique values:", np.unique(subseg))
        print("✅ subseg max value:", np.max(subseg))
        print("✅ subseg voxel sum:", np.sum(subseg))
        #print("subseg: ", subseg)
        if np.max(subseg) == 0:
            print(f"[ERROR] Subsegmentation failed for {in_img_file}. Skipping.")
            #return metadata
        #print("LOOOOOOOOOOOOOOOOOK4 seems good subseg") #REMOVE LATER 
        # Fourth stage - MSL
        print("before MSL")
        #print("Msl and after cause problems for some inputs - commented out for tests")
        #return metadata
        try:
            msl_p_planes = msl.find_planes(data_cropped, subseg)
            print("middle of MSL")
            msl_p_points = msl.findMSLforAllPlanes(data_cropped, subseg, msl_p_planes)
            print("after MSL")
            metadata["msl_planes"] = msl_p_planes
            metadata["msl_points"] = msl_p_points
            
            # Check if we have enough MSL results
            if len(msl_p_points) < 3:
                print(f"⚠️  MSL: Only {len(msl_p_points)} valid planes found. Pipeline may be incomplete.")
                
        except Exception as e:
            print(f"💥 MSL: Failed with error: {str(e)}")
            print("⚠️  MSL: Continuing with partial results...")
            # Set empty MSL results so pipeline can continue
            msl_p_planes = {}
            msl_p_points = {}
            metadata["msl_planes"] = msl_p_planes
            metadata["msl_points"] = msl_p_points

        # Fifth - Measuring
        print("\n===== DEBUG: About to look up BBD slice in MSL results =====")
        print("    sl_tcd_slice =", sl_tcd_slice)
        print("    sl_bbd_slice =", sl_bbd_slice)
        print("    len(msl_p_points)  =", len(msl_p_points))
        
        # Check if we have valid MSL results before proceeding
        if len(msl_p_points) == 0 or sl_bbd_slice not in msl_p_points:
            print(f"⚠️  MSL: Cannot perform measurements - MSL failed or slice {sl_bbd_slice} not available")
            print("⚠️  Saving partial results and exiting pipeline...")
            
            # Save what we have so far
            with open(os.path.join(out_dir, 'data.json'), 'w') as fp:
                json.dump(metadata, fp, cls=NumpyEncoder)
            print(f"💾 Partial results saved to: {out_dir}")
            return metadata
            
        print("    valid indices [0 ..", len(msl_p_points)-1, "]")
        # BBD + CBD
        CBD_min_th = (subseg.shape[0] / 4)
        p_u, p_d, _ = msl_p_points[sl_bbd_slice]
        CBD_measure, CBD_left, CBD_right = CBD_BBD.CBD_points(subseg[:, :, sl_bbd_slice] > 0,
                                                                   p_u.astype(int),
                                                                   p_d.astype(int), res_x, res_y,
                                                                  CBD_min_th)
        metadata["cbd_points"] = (CBD_left, CBD_right)
        metadata["cbd_measure_mm"] = CBD_measure
        BBD_measure, BBD_left, BBD_right, BBD_valid = CBD_BBD.BBD_points(data_cropped[:, :, sl_bbd_slice], CBD_left, CBD_right,
                                                             p_u.astype(int), p_d.astype(int),
                                                  res_x, res_y)
        metadata["bbd_points"] = (BBD_left, BBD_right)
        metadata["bbd_measure_mm"] = BBD_measure
        metadata["bbd_valid"] = BBD_valid

        # Professional measurement visualization plots
        def create_professional_measurement_plot(image_data, points_left, points_right, 
                                                measure_name, color, filename):
            plt.figure(figsize=(6, 6))
            plt.style.use('default')
            
            # Display the image with professional styling
            plt.imshow(image_data, cmap='gray', aspect='equal')
            
            # Plot measurement line with professional styling
            measurement_line = np.stack([points_left, points_right]).T
            plt.plot(measurement_line[1, :], measurement_line[0, :], 
                    color=color, linewidth=3, alpha=0.9, 
                    label=f'{measure_name} Measurement')
            
            # Add measurement points
            plt.scatter([points_left[1], points_right[1]], 
                       [points_left[0], points_right[0]],
                       color=color, s=60, edgecolor='white', 
                       linewidth=2, zorder=10)
            
            # Professional styling
            plt.title(f'{measure_name} Measurement Visualization', 
                     fontsize=14, fontweight='bold', 
                     color='#1f4e79', pad=15)
            
            plt.axis('off')  # Remove axes for cleaner look
            
            # Add legend
            legend = plt.legend(loc='upper right', fontsize=11, 
                               fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            
            plt.tight_layout()
            plt.savefig(os.path.normpath(os.path.join(out_dir, filename)), 
                       dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()

        # Create professional measurement plots
        create_professional_measurement_plot(data_cropped[:, :, sl_bbd_slice], 
                                           CBD_left, CBD_right, 'CBD', 
                                           '#dc3545', 'cbd.png')
        
        create_professional_measurement_plot(data_cropped[:, :, sl_bbd_slice], 
                                           BBD_left, BBD_right, 'BBD', 
                                           '#4a90e2', 'bbd.png')

        # TCD
        p_u, p_d, _ = msl_p_points[sl_tcd_slice]
        TCD_measure, TCD_left, TCD_right, TCD_valid = CBD_BBD.TCD_points(subseg[:, :, sl_tcd_slice] == 2., p_u.astype(int), p_d.astype(int),
                                                 res_x, res_y)
        metadata["tcd_valid"] = TCD_valid
        metadata["tcd_points"] = (TCD_left, TCD_right)
        metadata["tcd_measure_mm"] = TCD_measure
        
        # Create professional TCD plot
        if TCD_measure is not None:
            create_professional_measurement_plot(data_cropped[:, :, sl_tcd_slice], 
                                               TCD_left, TCD_right, 'TCD', 
                                               '#28a745', 'tcd.png')
        else:
            # Create placeholder plot if TCD measurement failed
            plt.figure(figsize=(6, 6))
            plt.imshow(data_cropped[:, :, sl_tcd_slice], cmap='gray', aspect='equal')
            plt.title('TCD Measurement - Not Available', 
                     fontsize=14, fontweight='bold', 
                     color='#dc3545', pad=15)
            plt.axis('off')
            plt.text(0.5, 0.5, 'TCD measurement\ncould not be determined', 
                    transform=plt.gca().transAxes,
                    ha='center', va='center',
                    fontsize=12, color='#dc3545',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            plt.tight_layout()
            plt.savefig(os.path.normpath(os.path.join(out_dir, "tcd.png")), 
                       dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
        
        from fetal_normative import predict_ga_from_measurement

        metadata["pred_ga_cbd"] = predict_ga_from_measurement(metadata["cbd_measure_mm"], "CBD")
        metadata["pred_ga_bbd"] = predict_ga_from_measurement(metadata["bbd_measure_mm"], "BBD")
        metadata["pred_ga_tcd"] = predict_ga_from_measurement(metadata["tcd_measure_mm"], "TCD")
        # Brain Volume Calc

        metadata["brain_vol_voxels"] = float(np.sum(fullseg > .5))
        metadata["brain_vol_mm3"] = float(np.sum(fullseg > .5) * res_x * res_y * res_z)
        # Build a short summary based on metadata
        summary = {
            "brain_volume_mm3": builtins.round(float(metadata["brain_vol_mm3"])),
            "CBD (mm)": builtins.round(float(metadata.get("cbd_measure_mm", 0))),
            "BBD (mm)": builtins.round(float(metadata.get("bbd_measure_mm", 0))),
            "TCD (mm)": builtins.round(float(metadata.get("tcd_measure_mm", 0))),
            "TCD valid": bool(metadata.get("tcd_valid", False)),
            "BBD valid": bool(metadata.get("bbd_valid", False)),
            "Volume path": metadata["InFile"]
        }

        # Append summary to metadata
        metadata["summary"] = summary
        # Dump metadata
        with open(os.path.normpath(os.path.join(out_dir, 'data.json')), 'w') as fp:
            json.dump(metadata, fp, cls=NumpyEncoder,indent=4)

        # --- Set gestational age week (default 30 for now, will be taken from metadata in future) ---
        GA_week = int(metadata.get('GA_week', 30))
        ga_source_note = ""
        if 'GA_week' not in metadata:
            ga_source_note = " (default)"
        
        # --- Get measurement values ---
        measured_dict = {
            'CBD': float(metadata.get("cbd_measure_mm", 0)),
            'BBD': float(metadata.get("bbd_measure_mm", 0)),
            'TCD': float(metadata.get("tcd_measure_mm", 0))
        }

        # --- Generate normative plots and statistics ---
        norm_results = normative_report_all(measured_dict, GA_week, out_dir)
        # Now norm_results['CBD']['plot_path'] etc. are ready for PDF use


        # ---- begin PDF report generation (with GA predictions inline) ----

        report_path = os.path.join(out_dir, 'report.pdf')
        with PdfPages(report_path) as pdf:
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor('white')
            
            # Create a more sophisticated grid layout
            gs = gridspec.GridSpec(6, 4, figure=fig,
                                    height_ratios=[0.5, 0.2, 1.2, 1.2, 1.2, 0.5],
                                    width_ratios=[1, 1, 1, 1],
                                    hspace=0.3, wspace=0.2,
                                    left=0.08, right=0.92, top=0.92, bottom=0.08)

            # Professional Header Section
            ax_header = fig.add_subplot(gs[0, :])
            ax_header.axis('off')
            
            # Main title with professional styling
            ax_header.text(0.5, 0.85, 'FETAL BRAIN MEASUREMENTS', 
                          ha='center', va='center',
                          fontsize=24, fontweight='bold', 
                          color='#1f4e79', family='serif')
            
            # Subtitle
            ax_header.text(0.5, 0.45, 'Automated Analysis Report', 
                          ha='center', va='center',
                          fontsize=14, style='italic',
                          color='#4a4a4a', family='serif')
            
            # Add a professional line separator
            ax_header.axhline(y=0.1, xmin=0.1, xmax=0.9, color='#1f4e79', linewidth=2)

            # Patient Information Section
            ax_info = fig.add_subplot(gs[1, :])
            ax_info.axis('off')
            
            pid = int(metadata['SubjectID'])
            ser = int(metadata['Series'])
            rx, ry, rz = metadata['Resolution']
            
            # Create professional info boxes
            info_text = f"""Patient ID: {pid}    •    Series: {ser}    •    Resolution: {rx:.3f}×{ry:.3f}×{rz:.3f} mm    •    GA: {GA_week}w{ga_source_note}"""
            
            ax_info.text(0.5, 0.5, info_text,
                        ha='center', va='center',
                        fontsize=11, color='#333333',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#f8f9fa', edgecolor='#dee2e6'))

            # Professional color scheme
            colors = {
                'primary': '#1f4e79',
                'secondary': '#4a90e2', 
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545',
                'light': '#f8f9fa',
                'dark': '#343a40'
            }

            def get_status_color(status):
                if status == "Normal":
                    return colors['success']
                elif status == "Below Norm":
                    return colors['warning']
                else:
                    return colors['danger']

            def create_measurement_row(row_idx, measure_name, image_file, value, pred_ga, 
                                     plot_path, status, slice_num, validity=True):
                # Image section
                ax_img = fig.add_subplot(gs[row_idx + 2, 0])
                img = plt.imread(os.path.join(out_dir, image_file))
                ax_img.imshow(img)
                ax_img.axis('off')
                ax_img.set_title(f'{measure_name} Measurement', fontsize=11, fontweight='bold', 
                               color=colors['primary'], pad=10)
                
                # Add professional border around image
                for spine in ax_img.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#dee2e6')
                    spine.set_linewidth(1)

                # Measurement details section
                ax_details = fig.add_subplot(gs[row_idx + 2, 1])
                ax_details.axis('off')
                
                # Create professional measurement card
                details_text = f"""
{measure_name} Measurement
{'─' * 15}
Value: {value:.2f} mm
Predicted GA: {pred_ga} weeks
Slice: #{slice_num}
Status: {status}
Valid: {'Yes' if validity else 'No'}
"""
                
                ax_details.text(0.05, 0.5, details_text,
                              ha='left', va='center',
                              fontsize=10, family='monospace',
                              bbox=dict(boxstyle="round,pad=0.5", 
                                      facecolor=colors['light'], 
                                      edgecolor='#dee2e6',
                                      linewidth=1))

                # Normative plot section
                ax_plot = fig.add_subplot(gs[row_idx + 2, 2:])
                plot_img = plt.imread(plot_path)
                ax_plot.imshow(plot_img)
                ax_plot.axis('off')
                
                # Status indicator with professional styling
                status_display = {"Below Norm": "Below Normal Range", 
                                "Above Norm": "Above Normal Range", 
                                "Normal": "Within Normal Range"}[status]
                
                ax_plot.set_title(f'Normative Analysis: {status_display}',
                                fontsize=11, fontweight='bold',
                                color=get_status_color(status), pad=10)
                
                # Add border around plot
                for spine in ax_plot.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#dee2e6')
                    spine.set_linewidth(1)

            # Create measurement rows
            create_measurement_row(0, 'CBD', 'cbd.png', 
                                 metadata['cbd_measure_mm'], metadata['pred_ga_cbd'],
                                 norm_results['CBD']['plot_path'], norm_results['CBD']['status'],
                                 metadata['BBD_selection'])
            
            create_measurement_row(1, 'BBD', 'bbd.png',
                                 metadata['bbd_measure_mm'], metadata['pred_ga_bbd'],
                                 norm_results['BBD']['plot_path'], norm_results['BBD']['status'],
                                 metadata['BBD_selection'], metadata['bbd_valid'])
            
            create_measurement_row(2, 'TCD', 'tcd.png',
                                 metadata['tcd_measure_mm'], metadata['pred_ga_tcd'],
                                 norm_results['TCD']['plot_path'], norm_results['TCD']['status'],
                                 metadata['TCD_selection'], metadata['tcd_valid'])

            # Professional Summary Section
            ax_summary = fig.add_subplot(gs[5, :])
            ax_summary.axis('off')
            
            # Add summary box with professional styling
            summary_text = f"""SUMMARY STATISTICS
Brain Volume: {metadata['brain_vol_mm3']:.0f} mm³  •  Total Voxels: {int(metadata['brain_vol_voxels'])}
TCD Slice Selection: {'Valid' if metadata['TCD_selectionValid'] else 'Invalid'}  •  BBD Slice Selection: {'Valid' if metadata['BBD_selectionValid'] else 'Invalid'}"""
            
            ax_summary.text(0.5, 0.9, summary_text,
                          ha='center', va='center',
                          fontsize=10, color='#333333',
                          bbox=dict(boxstyle="round,pad=0.4", 
                                  facecolor='#e8f4f8', 
                                  edgecolor=colors['primary'],
                                  linewidth=1))
            
            # Professional disclaimer - positioned lower to avoid overlap
            disclaimer = """This automated analysis is for research purposes only. Clinical decisions should not be based solely on these measurements.
Please consult with qualified medical professionals for clinical interpretation."""
            
            ax_summary.text(0.5, 0.3, disclaimer,
                          ha='center', va='center',
                          fontsize=8, style='italic', color='#666666',
                          bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor='#f8f9fa', 
                                  edgecolor='#dee2e6',
                                  alpha=0.8))

            # Add professional footer line
            ax_summary.axhline(y=0.05, xmin=0.1, xmax=0.9, color='#dee2e6', linewidth=1)

            pdf.savefig(fig, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
        # ---- end PDF report generation ----


        return metadata