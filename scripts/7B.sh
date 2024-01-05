export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
#--compile_prefill --compile
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth \
    --max_new_tokens $MAX_NEW_TOKENS --num_samples 1  --temperature 0.0 --batch_size $BATCH_SIZE --compile
