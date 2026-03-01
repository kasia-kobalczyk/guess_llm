from guess_llm.llm_utils.serialize import serialize_arr
import numpy as np
import torch
from tqdm import tqdm
from jax import vmap, grad


def llama_nll_fn(model, tokenizer, input_arr, target_arr, settings, transform):
    """Returns the NLL/dimension (log base e) of the target array (continuous) according to the LM
        conditioned on the input array. Applies relevant log determinant for transforms and
        converts from discrete NLL of the LLM to continuous by assuming uniform within the bins.
    inputs:
        input_arr: (n,) context array
        target_arr: (n,) ground truth array
        cache_model: whether to cache the model and tokenizer for faster repeated calls
    Returns: NLL/D
    """
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    full_series = input_str + target_str

    batch = tokenizer([full_series], return_tensors="pt", add_special_tokens=True)
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
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
    input_len = input_len - 2  # remove the BOS token

    logprobs = logprobs[input_len:]
    tokens = tokens[input_len:]
    BPD = -logprobs.sum() / len(target_arr)

    # print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec * np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    adjusted_nll = transformed_nll - avg_logdet_dydx

    return {
        "input_str": input_str,
        "target_str": target_str,
        "target_logprobs": logprobs,
        "decoded_target_tokens": tokens,
        "adjusted_nll": adjusted_nll,
    }


def get_hidden_states(model, tokenizer, input_arr, settings, transform):
    """
    Extract the hidden states from all layers of the model for the input array.
    """
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    batch = tokenizer([input_str], return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True)
    hidden_states = [h.cpu().numpy()[0, -1, :] for h in outputs.hidden_states]
    return hidden_states


def llama_completion_fn(
    model,
    tokenizer,
    input_str,
    steps,
    settings,
    batch_size=5,
    do_sample=True,
    num_samples=20,
    temp=0.9,
    top_p=0.9,
    cache_model=True,
):
    avg_tokens_per_step = len(tokenizer(input_str)["input_ids"]) / len(
        input_str.split(settings.time_sep)
    )
    max_tokens = int(avg_tokens_per_step * steps)

    gen_strs = []
    for _ in range(num_samples // batch_size):
        batch = tokenizer(
            [input_str],
            return_tensors="pt",
        )

        batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
        batch = {k: v.cuda() for k, v in batch.items()}
        num_input_ids = batch["input_ids"].shape[1]

        good_tokens_str = list("0123456789" + settings.time_sep)
        good_tokens = [
            tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str
        ]
        # good_tokens += [tokenizer.eos_token_id]
        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
        if do_sample:
            generate_ids = model.generate(
                **batch,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                max_new_tokens=max_tokens,
                bad_words_ids=[[t] for t in bad_tokens],
                renormalize_logits=True,
            )
        else:
            generate_ids = model.generate(
                **batch,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_tokens,
                bad_words_ids=[[t] for t in bad_tokens],
                renormalize_logits=True,
            )
        gen_strs += tokenizer.batch_decode(
            generate_ids[:, num_input_ids:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return gen_strs
