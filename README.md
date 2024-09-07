# Topofit w/ One hot Encoded Segmentation Map

To download the ground truths: use the following `wget` command:

```bash
wget https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis-wm-surfaces.v1.0.tar
```
Ensure the ground truths are saved as path/gt/{subject}/{hemi}.white.surf where {hemi} is lh/rh.

Run one_hot.py while giving paths of the Neurite Oasis dataset containing the 3D segmentation maps. Use seg4.nii.gz (5 channels), seg24.nii.gz (25 channels) or seg35.nii.gz (36 channels) as per requirement.
Save the one hot encodings as path/seg_files/{subject}/seg.npy

Run train.py like:
```bash
python train.py -o /local/scratch/v_karthik_mohan/output -t /local/scratch/v_karthik_mohan/topofit/train.txt -v /local/scratch/v_karthik_mohan/topofit/validation.txt --hemi lh
```
where -o is the path for storing output, -t is the path to train.txt which is a text file containing paths of all the subjects you are using for training. In this case, it would be like: path/seg_files/{subject}.
Use one line for each subject. -v is the path to validation.txt where you add paths of 5-20 subjects used for validation.

If you are modifying the input resolution, you need to make changes in io.py and model.py. Train.py will remain unchanged.

Similar to topofit, download the file neighbourhoods.npz and store it in the topofit subdirectory.


Suggested Environment: Python 3.10.14, PyTorch 1.11.0 compiled with CUDA 11.5
