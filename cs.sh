python -m mug_bench \
    --label "Skin Quality" \
    --dataset cs \
    --dataset_dir /home/felix.wernlein/project/datasets/CSGO-Skin-quality/ \
    --modality all \
    --hidden_size 256 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_epochs 500 \
    --patience 15 \
    --iterations 5 \
    --robust_training all \
    --missing_training \
    --missing_rate 1.0 \
    --missing_modality image tab \
    --contrastive_learning \
    --learning_rate_contrastive 4e-4\
    --batch_size_contrastive 128 \
    --num_epochs_contrastive 500 \
    --patience_contrastive 10 \
    --top_k 5 \