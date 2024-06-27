python inference.py \
    --cfg "/oceanstor/scratch/wohland/Current Members/shaoren/phd_files/miscellaneous_analysis/20240618_minimal_repo/imfcsnet-pytorch/workdirs/model_dextran_3d/config.yaml" \
    --ckpt "/oceanstor/scratch/wohland/Current Members/shaoren/phd_files/miscellaneous_analysis/20240618_minimal_repo/imfcsnet-pytorch/workdirs/model_dextran_3d/checkpoints/last.ckpt" \
    --files "/oceanstor/scratch/wohland/Current Members/Kulkarni/Rutuparna/InVi SPIM/Alginate_Sieving/TRITC_Dextran_Solution_Measurements/155kDa/*.tif" \
    --output-folder "model_outputs" \
    --device "cuda:0" \
    --bc-order 4