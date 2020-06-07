python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    train_fcos.py\
    --config-file configs/fcos/fcos_imprv_dcnv2_R_50_FPN_1x.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR training_dir/fcos_imprv_dcnv2_R_50_FPN_1x