python -m mug_bench \
    --label race \
    --dataset minion \
    --dataset_dir /home/felix.wernlein/project/datasets/Hearthstone-Minion-race/ \
    --modality all  \
    --hidden_size 256 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --num_epochs 500 \
    --patience 15 \
    --iterations 5 \
    --robust_training all \
    --missing_training \
    --missing_rate 1.0 \
    --missing_modality text image \
    --contrastive_learning \
    --learning_rate_contrastive 9e-5\
    --batch_size_contrastive 128 \
    --num_epochs_contrastive 500 \
    --patience_contrastive 5 \
    --top_k 4 \