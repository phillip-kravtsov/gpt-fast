export DRAFT_MODEL_REPO=philkrav/mistral-instruct-tinyllama-draft
export MODEL_REPO=mistralai/Mistral-7B-Instruct-v0.1/
python generate.py \
--draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth \
--checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth \
    --speculate_k 5 --max_new_tokens $MAX_NEW_TOKENS --num_samples 1  --temperature 0.0 --batch_size $BATCH_SIZE --compile
