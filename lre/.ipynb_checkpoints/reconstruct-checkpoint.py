import torch
import sys
sys.path.append('../..')
from transformers import GPTJForCausalLM, AutoTokenizer
import lre.models as models
import lre.functional as functional
import os

from baukit.baukit import parameter_names, get_parameter

device = "cuda:1"

#returns weight and bias for lns
def get_layer_norm_params(model, start, end):
    layer_norm_params = {}
    for i in range(start, end):
        w_name = f'transformer.h.{i}.ln_1.weight'
        b_name = f'transformer.h.{i}.ln_1.bias'
        weight = get_parameter(model=model,name=w_name).data.to(device)
        bias = get_parameter(model=model,name=b_name).data.to(device)
        layer_norm_params[w_name] = weight.to(device)
        layer_norm_params[b_name] = bias.to(device)
    return layer_norm_params

#Given a subject, kind (s_s, s_o, o_o) and layer
#Returns the weight and bias for that subject alone.
def sample_weights_biases(subject, kind, i) -> dict:
    layer_dict = {"i": i}
    weights = []
    biases = []
    wdir = subject
    weight_path = f"{wdir}/{i}/{kind}_weight_h_{i}.pt"
    bias_path = f"{wdir}/{i}/{kind}_bias_h_{i}.pt"
    #load s_s_weight and s_s_bias
    weight = torch.load(weight_path).to(device)
    bias = torch.load(bias_path).to(device)
    layer_dict[f'{kind}_weight'] = weight
    layer_dict[f'{kind}_bias'] = bias
    return layer_dict

#Given a kind (s_s, s_o, o_o), layer, and samples
#Returns a dictionary with the mean weight and bias

def mean_weights_biases(kind, i, samples) -> dict:
    layer_dict = {"i": i}
    weights = []
    biases = []
    for sample in samples:
        wdir = sample
        weight_path = f"{wdir}/{i}/{kind}_weight_h_{i}.pt"
        bias_path = f"{wdir}/{i}/{kind}_bias_h_{i}.pt"
        #load s_s_weight and s_s_bias
        weight = torch.load(weight_path).to(device)
        bias = torch.load(bias_path).to(device)
        #print(f'weight is {tp(weight)}')
        #append to lists
        weights.append(weight)
        biases.append(bias)
    mean_weight = torch.stack(weights).mean(dim=0).to(device)
    mean_bias = torch.stack(biases).mean(dim=0).to(device)
    sum_weight = torch.stack(weights).sum(dim=0).to(device)
    sum_bias = torch.stack(biases).sum(dim=0).to(device)
    
    layer_dict[f'{kind}_mean_weight'] = mean_weight
    layer_dict[f'{kind}_mean_bias'] = mean_bias
    layer_dict[f'{kind}_sum_weight'] = sum_weight
    layer_dict[f'{kind}_sum_bias'] = sum_bias
    
    return layer_dict

#Given an output hidden state
#Returns the lm's next token prediction
#lm_head applies LayerNorm and then a linear map to get the token-space (50400)
# (1,4096) -layernorm, linear-> (1,50400) -softmax-> (1,50400) -topk-> (1,5)
def get_object(mt, z, k=5):
    logits = mt.lm_head(z)
    dist = torch.softmax(logits.float(), dim=-1)
    topk = dist.topk(k=k, dim=-1)
    probs = topk.values.view(5).tolist()
    token_ids = topk.indices.view(5).tolist()
    words = [mt.tokenizer.decode(token_id) for token_id in token_ids]
    return (words, probs)


def layer_norm(
    x: torch.Tensor, dim, eps: float = 0.00001
) -> torch.Tensor:
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.square(x - mean).mean(dim=dim, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)

def approx_s_s_layer(hs, i, layer_dict, params):
    layer_weight = layer_dict['s_s_mean_weight']
    layer_bias = layer_dict['s_s_mean_bias']
    ln_weight = params[f'transformer.h.{i}.ln_1.weight']
    ln_bias = params[f'transformer.h.{i}.ln_1.bias']
    _hs = hs
    
    #perform layer normalization with adaptive w and b
    hs = layer_norm(hs, (1)) * ln_weight + ln_bias
    
    #perform the layer operation
    hs = hs.mm(layer_weight.t()) #+ layer_bias
        
    #add residual
    #hs = hs + _hs
    return hs

def approx_s_o_layer(hs, i, layer_dict, params):
    layer_weight = layer_dict['s_o_mean_weight']
    layer_bias = layer_dict['s_o_mean_bias']  
    ln_weight = params[f'transformer.h.{i}.ln_1.weight']
    ln_bias = params[f'transformer.h.{i}.ln_1.bias']
    _hs = hs
    
    #perform layer normalization with adaptive w and b
    hs = layer_norm(hs, (1)) * ln_weight + ln_bias
    
    #perform the layer operation
    hs = hs.mm(layer_weight.t()) + layer_bias
    
    #add residual
    hs = hs + _hs * 2.5
    return hs
    
def approx_o_o_layer(hs, i, layer_dict, params):
    layer_weight = layer_dict['o_o_mean_weight']
    layer_bias = layer_dict['o_o_mean_bias']  
    ln_weight = params[f'transformer.h.{i-1}.ln_1.weight']
    ln_bias = params[f'transformer.h.{i-1}.ln_1.bias']
    _hs = hs
    
    #perform layer normalization with adaptive w and b
    hs = layer_norm(hs, (1)) * ln_weight + ln_bias
    
    #perform the layer operation
    hs = hs.mm(layer_weight.t()) + layer_bias
    
    #add residual
    hs = hs + _hs
    return hs
    
def get_hidden_state(mt, subject, h_layer):
    prompt = f"The offspring of a {subject} is referred to as a"
    h_index, inputs = functional.find_subject_token_index(
        mt = mt, prompt=prompt, subject=subject)
    #print(f'h_index is {h_index}, inputs is {inputs}')
    [[hs], _] = functional.compute_hidden_states(
        mt = mt, layers = [h_layer], inputs = inputs)
    #h is hs @ h_layer @ h_index
    h = hs[:, h_index]
    h = h.to(device)
    return h