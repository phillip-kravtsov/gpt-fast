# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import json
import numpy as np

import torch
import torch._dynamo.config
import torch._inductor.config

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

from model import Transformer
from tp import maybe_init_dist

PAD_TOKEN_ID = 1
REPORT_PATH = "output.jsonl"
RECORD_EVENTS = True


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    model_time = []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
<<<<<<< HEAD
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs
=======
            end_event.record()
            if RECORD_EVENTS:
                torch.cuda.synchronize()
                model_time.append(start_event.elapsed_time(end_event))
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.view(cur_token.size(0), -1)
    return new_tokens, new_probs, model_time
>>>>>>> 86b0d1e3b367482e0ffea2b19dcf5690134f6096


def model_forward(model, x, input_pos, batch_index):
    return model(x, input_pos, batch_index)


def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    speculate_k: int,
    **sampling_kwargs,
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    batch_size = cur_token.size(0)
    orig_input_pos = input_pos.clone().to(cur_token.device)
    draft_tokens, draft_probs, _ = decode_n_tokens(
        draft_model,
        cur_token.view(batch_size, -1),
        orig_input_pos.clone(),
        speculate_k,
        **sampling_kwargs,
    )

    draft_tokens = torch.cat(draft_tokens, dim=1)
    # parallel inference on target model using draft tokens
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    draft_token_inputs = torch.cat(
        [cur_token.view(batch_size, 1), draft_tokens], dim=1
    ).view(batch_size, -1)
    input_pos_inputs = input_pos + torch.arange(
        speculate_k + 1, device=input_pos.device
    )

    start_event.record()
    target_logits = model_forward(
        model,
        draft_token_inputs,
        input_pos_inputs,
        None,
    )
    end_event.record()
    speculate_time = 0.0
    if RECORD_EVENTS:
        torch.cuda.synchronize()
        speculate_time = start_event.elapsed_time(end_event)

    target_probs = logits_to_probs(target_logits, **sampling_kwargs)
    draft_probs = torch.stack(draft_probs, dim=1)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    batch_indices = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, speculate_k)
    )
    sequence_indices = (
        torch.arange(speculate_k, device=device).unsqueeze(0).expand(batch_size, -1)
    )

    p = draft_probs[batch_indices, sequence_indices, draft_tokens]
    q = target_probs[batch_indices, sequence_indices, draft_tokens]

    accept_draft_prob = torch.minimum(torch.ones(()), q / p)
    sequences = []
    for i in range(batch_size):
        rejected_locations = (
            torch.rand_like(accept_draft_prob[i]) > accept_draft_prob[i]
        ).nonzero()
        if rejected_locations.shape[0] == 0:
            accept_length = speculate_k + 1
            last_token = multinomial_sample_one_no_sync(target_probs[i, -1])
            model_forward(
                draft_model,
                draft_tokens[i, -1].view(1, -1),
                orig_input_pos[i : i + 1] + speculate_k,
                i,
            )
            sequences.append(torch.cat([draft_tokens[i], last_token]))
        else:
            accept_length = rejected_locations[0].item()
            p = draft_probs[i, accept_length]
            q = target_probs[i, accept_length]
            new = q - p
            new = torch.where(new > 0, new, 0.0)
            new = new / new.sum()
            next_token = multinomial_sample_one_no_sync(new)
            sequences.append(torch.cat([draft_tokens[i, :accept_length], next_token]))
    return sequences, speculate_time


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback=lambda x: x,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    print("Generating")
    prompt = prompt.repeat(batch_size, 1)
    T = prompt.size(1)

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = (
        max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    )
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(
                max_batch_size=batch_size, max_seq_length=max_seq_length
            )

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty((batch_size, T_new), dtype=dtype, device=device)
    seq = empty
    seq[:, :T] = prompt

    input_pos = torch.arange(0, T, device=device).repeat(batch_size, 1)
    prefill_start = time.perf_counter()
    next_token = prefill(
        model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs
    )

    if is_speculative:
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    prefill_time = time.perf_counter() - prefill_start
    seq[:, T] = next_token.view(-1)

    input_pos = torch.tensor([T], device=device, dtype=torch.int).repeat(batch_size, 1)
    accept_counts = [0] * (speculate_k + 1)
    speculative_times = []
    model_times = []

    if is_speculative:
        while max(input_pos) < T_new - 1:
            cur_token = next_token.view(batch_size)
            next_token_sequences, speculative_time = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )
            speculative_times.append(speculative_time)
            for i in range(batch_size):
                accept_counts[next_token_sequences[i].size(0) - 1] += 1
            # Update sequences with the new tokens, each of which may have variable sizes.
            for i, sequence in enumerate(next_token_sequences):
                num_added = min(T_new - input_pos[i] - 1, len(sequence))
                seq[i, input_pos[i] + 1 : input_pos[i] + num_added + 1] = sequence[
                    :num_added
                ]
                input_pos[i] += num_added
            next_token = torch.cat(
                [sequence[-1].unsqueeze(0) for sequence in next_token_sequences]
            )
    else:
        generated_tokens, _, model_times = decode_n_tokens(
            model,
            next_token.view(batch_size, -1),
            input_pos,
            max_new_tokens - 1,
            callback=callback,
            **sampling_kwargs,
        )
        seq[:, T + 1 :] = torch.cat(generated_tokens, dim=1)

    generate_stats = {
        "accept_counts": accept_counts,
        "prefill_time": prefill_time,
        "speculative_times": speculative_times,
        "model_times": model_times,
    }
    trimmed_sequences = []
    for s, ip in zip(seq.tolist(), input_pos.tolist()):
        trimmed_sequences.append(s[: ip[0]])
    return trimmed_sequences, generate_stats


def encode_tokens(tokenizer, string, bos=True, device="cuda"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision, use_tp):
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp

        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    device='cuda',
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

<<<<<<< HEAD
    print(f"Using device={device}")
=======
    device = "cuda:0"
>>>>>>> 86b0d1e3b367482e0ffea2b19dcf5690134f6096
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        print("Loading draft model.")
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    prompt_length = encoded.size(0)

    torch.manual_seed(1234)
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )
    if compile:
<<<<<<< HEAD
        if is_speculative and use_tp: # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case
=======
        print("Compiling model.")
        fullgraph = True
        # torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.cache_size_limit = 512
        if is_speculative and use_tp:
            torch._inductor.config.triton.cudagraph_trees = (
                False  # Bug with cudagraph trees in this case
            )
>>>>>>> 86b0d1e3b367482e0ffea2b19dcf5690134f6096

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(
                model_forward, mode="reduce-overhead", fullgraph=fullgraph
            )

        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=fullgraph
        )

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            print("Compiling prefill")
            prefill = torch.compile(prefill, fullgraph=fullgraph, dynamic=True)

    aggregate_metrics = {
        "tokens_per_sec": [],
        "accept_counts": [],
        "tokens_generated": [],
        "prefill_time": [],
        "speculative_times": [],
        "model_times": [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device) # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
        else:
            callback = lambda x: x
        t0 = time.perf_counter()
        import contextlib

        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        if not interactive:
            sequences = tokenizer.decode(y)
            for sequence in sequences:
                print(sequence)
        else:
            for y_i in y:
                print(tokenizer.decode(y_i))

        aggregate_metrics["accept_counts"].append(metrics["accept_counts"])
        aggregate_metrics["prefill_time"].append(metrics["prefill_time"])
        aggregate_metrics["speculative_times"].extend(metrics["speculative_times"])
        aggregate_metrics["model_times"].extend(metrics["model_times"])
        tokens_generated = sum(len(y_i) - prompt_length for y_i in y)
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_generated"].append(tokens_generated)
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

    print("==========")
    avg_tokens_generated = sum(aggregate_metrics["tokens_generated"]) / num_samples
    print(f"Average tokens generated: {avg_tokens_generated}")

    report = {
        "t": t,
        "batch_size": batch_size,
        "compile": compile,
        "is_speculative": is_speculative,
        "tokens_generated": avg_tokens_generated,
        "num_samples": num_samples,
        "model": str(checkpoint_path),
        "draft_model": str(draft_checkpoint_path),
    }

    if is_speculative:
        print(aggregate_metrics["accept_counts"])
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics["accept_counts"])]
        print(counts_aggregated)
        acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        mean_accepted = sum([idx * i for idx, i in enumerate(counts_aggregated)]) / sum(
            counts_aggregated
        )
        print(f"Mean Accepted: {mean_accepted} tokens")
        report["mean_accepted"] = mean_accepted
        report["acceptance_rate"] = mean_accepted / speculate_k
        report["speculative_time_mean"] = np.array(
            aggregate_metrics["speculative_times"]
        ).mean()
    else:
        report["model_time_mean"] = np.array(aggregate_metrics["model_times"]).mean()

    tok_s = torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])).item()
    user_tok_s = (
        torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])) / batch_size
    ).item()
    report["tok_s"] = tok_s
    report["user_tok_s"] = user_tok_s

    with open(REPORT_PATH, "a") as f:
        json.dump(report, f)
        f.write("\n")

    print(f"Average tokens/sec: {tok_s:.2f}")
    print(f"User tok/sec: {user_tok_s:.2f}")
    print(
        f"Average prefill time: {torch.mean(torch.tensor(aggregate_metrics['prefill_time'])).item():.2f}"
    )
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    import argparse

<<<<<<< HEAD
    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--device', type=str, default="cuda", help='device to use')

    args = parser.parse_args()
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.profile, args.draft_checkpoint_path,
        args.speculate_k, args.device
=======
    parser = argparse.ArgumentParser(description="Your CLI description.")
    prompt = """
    Here's some irrelevant info:
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits, medusa_logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - medusa_logits (torch.Tensor): Updated medusa logits.
    - new_token (int): Updated counter for the new tokens added.

    Now ignore that and write a quicksort in C++ three times in a row.
    """
    prompt = "<<SYS>>\nYou are an expert programmer\n<</SYS>>\n\n[INST] Write a quicksort in python.[/INST]"

    parser.add_argument(
        "--prompt",
        type=str,
        default=prompt,
        help="Input prompt.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to launch in interactive mode",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile_prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--speculate_k", type=int, default=5, help="Speculative execution depth."
    )
    parser.add_argument(
        "--draft_checkpoint_path",
        type=Path,
        default=None,
        help="Draft checkpoint path.",
    )
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    main(
        prompt=args.prompt,
        interactive=args.interactive,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        batch_size=args.batch_size,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
        compile=args.compile,
        compile_prefill=args.compile_prefill,
        profile=args.profile,
        draft_checkpoint_path=args.draft_checkpoint_path,
        speculate_k=args.speculate_k,
>>>>>>> 86b0d1e3b367482e0ffea2b19dcf5690134f6096
    )
