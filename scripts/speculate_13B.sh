export DRAFT_MODEL_REPO=philkrav/tinyllama-1.3b-draft-llama-13b-chat
#export DRAFT_MODEL_REPO=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export MODEL_REPO=meta-llama/Llama-2-13b-chat-hf
python generate.py --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth \
    --speculate_k 5 --max_new_tokens $MAX_NEW_TOKENS --num_samples 1  --temperature 0.0 --batch_size $BATCH_SIZE \
    --prompts 'prompts.jsonl'
