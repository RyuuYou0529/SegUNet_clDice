python train_SegNet3D.py    -model SegNet3D_cldice \
                            -arch SegNet3D \
                            -dataset dataset_skels \
                            -trainer trainer_SegNet3D \
                            -data data/skels_aug \
                            -gpus 0 \
                            -batch_per_gpu 32 \
                            -epochs 2500 \
                            -save_every 10 \
                            -out out \
                            -cfg config/SegNet3D.yaml \
                            -slurm \
                            -slurm_ngpus 4 \
                            -slurm_nnodes 1 \
                            -slurm_nodelist c002 \
                            -slurm_partition compute \
                            # -reset \