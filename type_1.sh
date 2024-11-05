python -m mug_bench \
    --label type_1 \
    --dataset type_1 \
    --dataset_dir /home/felix.wernlein/project/datasets/Pokemon-primary_type/ \
    --modality all \
    --hidden_size 256 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --num_epochs 500 \
    --patience 15 \
    --iterations 5 \
    --contrastive_learning \
    --learning_rate_contrastive 2e-4\
    --batch_size_contrastive 64 \
    --num_epochs_contrastive 500 \
    --patience_contrastive 10 \
    --top_k 5 \
    #--robust_training all \
    #--missing_training \
    #--missing_rate 1.0 \
    #--missing_modality text image \
