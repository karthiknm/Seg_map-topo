import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom




base_dir = "/local/scratch/v_karthik_mohan/data"
output_dir = "/local/scratch/v_karthik_mohan/seg_files"
os.makedirs(output_dir, exist_ok=True)
i = 0

# Function to one-hot encode the segmentation map
def one_hot_encode(segmentation, num_classes=5):
    one_hot = np.eye(num_classes)[segmentation]
    return one_hot

# Rescale the volume to 36x36x36, change according to target resolution
def resize_volume(volume, target_shape=(36, 36, 36)):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=0)


for subject_dir in sorted(os.listdir(base_dir)):
    subject_path = os.path.join(base_dir, subject_dir, "seg4.nii.gz")
    print("Total number of files completed: {}".format(i))
    i += 1


    if os.path.isfile(subject_path):
        print(f"Processing {subject_path}...")
        try:
       
            seg_img = nib.load(subject_path)
            seg_data = seg_img.get_fdata()

            # Resize the segmentation to 36x36x36
            resized_seg = resize_volume(seg_data)

            # One-hot encode the resized segmentation
            one_hot_seg = one_hot_encode(resized_seg.astype(int))

            # Prepare output directory for the subject
            subject_name = subject_dir  # Extracting the subject name
            subject_output_dir = os.path.join(output_dir, subject_name)
            os.makedirs(subject_output_dir, exist_ok=True)  # Create the directory if it doesn't exist

            # Save the one-hot encoded segmentation as seg.npy file
            output_file = os.path.join(subject_output_dir, "seg.npy")
            np.save(output_file, one_hot_seg)
            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Error processing {subject_path}: {e}")
    else:
        print(f"File {subject_path} not found.")
