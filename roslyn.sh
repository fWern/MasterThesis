python -m muldic_bench \
    --label Roslyn \
    --dataset_dir /home/felix.wernlein/project/datasets/muldic/ \
    --modality all \
    --hidden_size 256 \
    --batch_size 64 \
    --learning_rate 9e-5 \
    --num_epochs 500 \
    --patience 10 \
    --iterations 5 \
    --robust_training all \
    --contrastive_learning \
    --learning_rate_contrastive 1.5e-5\
    --batch_size_contrastive 128 \
    --num_epochs_contrastive 500 \
    --patience_contrastive 5 \
    --top_k 5 \
    #--missing_training \
    #--missing_rate 1.0 \
    #--missing_modality image code \