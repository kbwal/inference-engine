from architecture import Qwen3_1_7B, make_qwen_1_7
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_weights import load_weights
from run_tokenization import encode, decode, gen_tokenizer

@torch.inference_mode()
def autoregressive_loop(model: Qwen3_1_7B, tokenizer, inputs: list[str],tau: float, max_num_tokens: int, device) -> list[str]:
    msgs = []
    for input in inputs:
        msgs.append([{"role": "user", "content": input}])
    token_ids, attention_masks = encode(msgs)
    token_ids = token_ids.to(device)
    attention_masks = attention_masks.to(device)
    res = []
    curr = torch.ones(len(inputs), dtype=torch.bool, device = device)
    for _ in range(max_num_tokens):
        raw = model(token_ids,attention_mask=attention_masks)
        probs = (raw[curr] / tau).softmax(-1)
        sampled_indices = torch.multinomial(probs, num_samples=1)
        sampled_indices = sampled_indices.squeeze(-1)
        next_ids = torch.full(size=(len(inputs),),fill_value=tokenizer.eos_token_id, dtype=torch.long, device=device)
        next_ids[curr] = sampled_indices
        res.append(next_ids)

        token_ids = torch.cat((token_ids, next_ids[:, None]), dim=-1)
        attention_masks = torch.cat([attention_masks,curr[:, None].to(attention_masks.dtype)],dim=-1)
        curr = curr & ~(curr & (next_ids == tokenizer.eos_token_id)) 
    output_ids = torch.stack(res, 1).to("cpu")
    return decode(output_ids)

if __name__ == "__main__":
    model = make_qwen_1_7()
    model.load_state_dict(load_weights(device="cpu"))
    res = autoregressive_loop(
        model=model, 
        tokenizer=gen_tokenizer(), 
        inputs = ["2*3=","What is your name?"],
        tau=0.3,
        max_num_tokens=32,
        device="mps"
        )
    print(res)




        
    
        
    