python -m sentiment_bench \
    --label acc7 \
    --dataset mosi \
    --dataset_dir /home/felix.wernlein/project/datasets/mosi/ \
    --modality all \
    --hidden_size 128 \
    --batch_size 128 \
    --learning_rate 3e-3 \
    --num_epochs 500 \
    --patience 20 \
    --iterations 5 \
    --contrastive_learning \
    --learning_rate_contrastive 8e-4\
    --batch_size_contrastive 64 \
    --num_epochs_contrastive 500 \
    --patience_contrastive 10 \
    --top_k 4 \
    --robust_training all \
    --missing_training \
    --missing_rate 1.0 \
    --missing_modality image audio \