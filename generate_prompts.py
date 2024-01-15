import transformers as tr
from datasets import load_dataset

ds = load_dataset('bigcode/humanevalpack')['test']
instructions = [item['instruction'] for item in ds]

tokenizer = tr.AutoTokenizer.from_pretrained(
    'philkrav/tinyllama-1.3b-draft-llama-13b-chat'
)

prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
) for instruction in instructions]

with open('prompts.jsonl', 'w') as f:
    import json
    for prompt in prompts:
        f.write(json.dumps({"prompt": prompt + '\n'}))
        f.write('\n')


