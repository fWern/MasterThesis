python -m sentiment_bench \
    --label acc7 \
    --dataset mosei \
    --dataset_dir /home/felix.wernlein/project/datasets/mosei/ \
    --modality all \
    --hidden_size 128 \
    --batch_size 128 \
    --learning_rate 9e-5 \
    --num_epochs 500 \
    --patience 1 \
    --iterations 2 \
    --contrastive_learning \
    --learning_rate_contrastive 1e-4\
    --batch_size_contrastive 128 \
    --num_epochs_contrastive 500 \
    --patience_contrastive 10 \
    --top_k 4 \
    --robust_training all \
    --missing_training \
    --missing_rate 0.9 \
    --missing_modality audio \