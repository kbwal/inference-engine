from architecture import Qwen3_1_7B, make_qwen_1_7
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_weights import load_weights
from run_tokenization import encode, decode

def _raw_output_dist(model, inputs: list[str])->torch.Tensor: 
    msgs = []
    for input in inputs:
        msgs.append([{"role": "user", "content": input}])
    token_ids, attention_mask = encode(msgs)
    raw = model.forward(token_ids.to("mps"), attention_mask=attention_mask.to("mps"))
    return raw

def batch_greedy_decode(model, inputs:list[str])->list[str]: 
    raw = _raw_output_dist(model, inputs)
    return decode(torch.argmax(raw,-1))

@torch.inference_mode()
def batch_temperature_sampling(model, inputs: list[str],tau)->list[str]:
    raw = _raw_output_dist(model, inputs)
    raw /= tau 
    dist = raw.softmax(-1, dtype= torch.bfloat16)
    sampled_indices = torch.multinomial(dist, num_samples=1)
    sampled_indices = sampled_indices.squeeze(-1)
    return decode(sampled_indices)

    
if __name__=="__main__":
    model = make_qwen_1_7()
    model.load_state_dict(load_weights(device="cpu"))
    print(batch_greedy_decode(model, ["3*2: "]))
    print(batch_temperature_sampling(model, ["adslf", "My name is "], 0.9))

