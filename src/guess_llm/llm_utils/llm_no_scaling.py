import torch
from tqdm import tqdm
import numpy as np
from jax import grad, vmap

GOOD_LLAMA2_TOKENS = [
    448,
    29889,
    29871,
    29892,
    29953,
    29955,
    29896,
    29929,
    29900,
    29906,
    29941,
    29945,
    29946,
    29947,
]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def serialize(arr, precision=3):
    str_ = ""
    for x in arr:
        str_ += f"{x:0.{precision}f}, "
    return str_[:-1]


def deserialize(str_):
    str_ = [(s.split(",")[0]) for s in str_]
    str_ = [s[:-1] if s[-1] == "." else s for s in str_]
    return [float(s) for s in str_ if is_number(s)]


def llm_completion_fn(
    model_name,
    model,
    tokenizer,
    input_str,
    steps,
    do_sample=True,
    num_samples=20,
    temp=1.0,
    top_p=1.0,
    cache_model=True,
):
    time_sep = ", "
    avg_tokens_per_step = len(tokenizer(input_str)["input_ids"]) / len(
        input_str.split(time_sep)
    )
    max_tokens_per_step = max(
        [
            len(input_str.split(time_sep)[i])
            for i in range(len(input_str.split(time_sep)))
        ]
    )
    max_tokens = int(max_tokens_per_step * steps)

    batch = tokenizer(
        [input_str],
        return_tensors="pt",
    )

    batch = {k: v.repeat(num_samples, 1) for k, v in batch.items()}
    batch = {k: v.to(model.device) for k, v in batch.items()}
    num_input_ids = batch["input_ids"].shape[1]

    if model_name == "meta-llama/Llama-2-7b-hf":
        # assuming llama2 tokenization
        good_tokens = GOOD_LLAMA2_TOKENS
        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
        generate_ids = model.generate(
            **batch,
            do_sample=do_sample,
            temperature=temp,
            top_p=top_p,
            max_new_tokens=max_tokens,
            bad_words_ids=[[t] for t in bad_tokens],
            renormalize_logits=True,
        )
    else:
        generate_ids = model.generate(
            **batch,
            do_sample=do_sample,
            temperature=temp,
            top_p=top_p,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_strs = tokenizer.batch_decode(
        generate_ids[:, num_input_ids:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return gen_strs


def llama_generate_samples(
    input_str,
    n_samples,
    model_name,
    model,
    tokenizer,
    do_sample=True,
    temp=1.0,
    top_p=1.0,
):
    num_samples_left = n_samples
    samples = []
    i = 0
    while num_samples_left > 0 and i < 5:
        gen_str = llm_completion_fn(
            model_name,
            model,
            tokenizer,
            input_str,
            steps=1,
            num_samples=int(num_samples_left * 1.1),
            do_sample=do_sample,
            temp=temp,
            top_p=top_p,
        )
        clean_samples = deserialize(gen_str)
        samples += clean_samples
        num_samples_left -= len(clean_samples)
        i += 1
    return samples[:n_samples]


def get_hidden_states_no_scaling(model, tokenizer, input_str):
    """
    Extract the hidden states from all layers of the model for the input array.
    """
    batch = tokenizer([input_str], return_tensors="pt")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True)
    hidden_states = [h.cpu().numpy()[0, -1, :] for h in outputs.hidden_states]
    return hidden_states


def llama_nll_no_scaling(model, tokenizer, input_str, target_str, precision=3):
    """Returns the NLL/dimension (log base e) of the target array (continuous) according to the LM
        conditioned on the input array. Applies relevant log determinant for transforms and
        converts from discrete NLL of the LLM to continuous by assuming uniform within the bins.
    inputs:
        input_arr: (n,) context array
        target_arr: (n,) ground truth array
        cache_model: whether to cache the model and tokenizer for faster repeated calls
    Returns: NLL/D
    """
    full_series = input_str + " " + target_str

    batch = tokenizer([full_series], return_tensors="pt", add_special_tokens=True)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    good_tokens = GOOD_LLAMA2_TOKENS
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    out["logits"][:, :, bad_tokens] = -100

    input_ids = batch["input_ids"][0][1:]
    logprobs = torch.nn.functional.log_softmax(out["logits"], dim=-1)[0][:-1]
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()

    tokens = tokenizer.batch_decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    input_len = len(
        tokenizer(
            [input_str],
            return_tensors="pt",
        )[
            "input_ids"
        ][0]
    )
    input_len = input_len - 1  # remove the BOS token

    logprobs = logprobs[input_len:]
    tokens = tokens[input_len:]
    BPD = -logprobs.sum()

    # print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - precision * np.log(10)
    # avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    # adjusted_nll = transformed_nll - avg_logdet_dydx

    return {
        "input_str": input_str,
        "target_str": target_str,
        "target_logprobs": logprobs,
        "decoded_target_tokens": tokens,
        "adjusted_nll": transformed_nll,
    }
