python -m muldic_bench \
    --label "VS Code" \
    --dataset_dir /home/felix.wernlein/project/datasets/muldic/ \
    --modality all \
    --hidden_size 256 \
    --batch_size 128 \
    --learning_rate 1.5e-4 \
    --num_epochs 500 \
    --patience 15 \
    --iterations 5 \
    --robust_training all \
    #--contrastive_learning \
    #--learning_rate_contrastive 2.5e-5\
    #--batch_size_contrastive 128 \
    #--num_epochs_contrastive 500 \
    #--patience_contrastive 5 \
    #--top_k 4 \
    #--missing_training \
    #--missing_rate 1.0 \
    #--missing_modality image code \