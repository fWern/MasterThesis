python -m mug_bench \
    --label Category \
    --dataset lol \
    --dataset_dir /home/felix.wernlein/project/datasets/LeagueOfLegends-Skin-category/ \
    --modality pcag \
    --hidden_size 128 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --num_epochs 500 \
    --patience 20 \
    --iterations 5 \
    --pcag_dim 150 \
    --pcag_batch_size 64 \
    --pcag_learning_rate 2e-4 \
    --contrastive_learning \
    --learning_rate_contrastive 3e-4\
    --batch_size_contrastive 32 \
    --num_epochs_contrastive 500 \
    --patience_contrastive 10 \
    --top_k 5 \
    #--robust_training all \
    #--missing_training \
    #--missing_rate 1.0 \
    #--missing_modality image tab \