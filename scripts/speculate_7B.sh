export DRAFT_MODEL_REPO=PY007/TinyLlama-1.1B-intermediate-step-480k-1T
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model.pth --checkpoint_path checkpoints/$MODEL_REPO/model.pth \
    --speculate_k 5 --max_new_tokens $MAX_NEW_TOKENS --num_samples 1  --temperature 0.0 --batch_size $BATCH_SIZE --compile
