#This script was used to produce test results for GPT-J.
#Given: folder names containing individual Jacobian and W2V (E(o - s)) tensors
#Given: device number, N_TRIALS, start & end layers for Jacobian, beta for LRE
#Builds LRE, Jacobian, Bias, and W2V approximators
#Tests each relation over N_TRIALS, returning the highest # correct for each method as well as the LM correct 

import torch
import sys
sys.path.append('..')
from transformers import GPTJForCausalLM, AutoTokenizer
import os
import json
import random

import lre.models as models
import lre.functional as functional
from lre.data import Relation, RelationSample, Sequence
import lre.metrics as metrics
import lre.functional as functional
import lre.logging_utils as logging_utils
import llra.build as build
import logging

from baukit.baukit import parameter_names, get_parameter
import torch.nn as nn

#MAKE SURE TO CHANGE THIS EACH RUN
DEVICE_NUM = 1
RESULTS_FILE = f'results/42Thursday_GPU2_Cuda{DEVICE_NUM}_3_9.txt'


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=RESULTS_FILE,
    level=logging.INFO,
    format = logging_utils.DEFAULT_FORMAT,
    datefmt=logging_utils.DEFAULT_DATEFMT,
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token

mt = models.ModelAndTokenizer(model,tokenizer)

device = f'cuda:{DEVICE_NUM}'
model.to(device)

build.determine_device(mt)
build.determine_params(mt)
end = 27

#OBTAIN LRE WEIGHTS
relations = [
 'name - nationality',
 'animal - youth',
 'verb_Ving - 3pSg',
 'noun+less_reg',
 'verb+able_reg',
 'UK_city - county',
 'antonyms - binary',
 'verb_inf - 3pSg',
 're+verb_reg',
 'verb_inf - Ved',
 'country - language',
 'meronyms - part',
 'verb_Ving - Ved',
 'animal - shelter',
 'hypernyms - misc',
 'meronyms - substance',
 'noun - plural_irreg',
 'un+adj_reg',
 'verb+ment_irreg',
 'adj+ness_reg',
 'over+adj_reg',
 'verb+er_irreg',
 'adj+ly_reg',
 'name - occupation',
 'synonyms - intensity',
 'animal - sound',
 'noun - plural_reg',
 'Ving - verb_inf',
 'male - female',
 'verb_3pSg - Ved',
 'meronyms - member',
 'things - color',
 'hyponyms - misc',
 'adj - superlative',
 #'.ipynb_checkpoints',
 'verb+tion_irreg',
 'synonyms - exact',
 'hypernyms - animals',
 'country - capital']

#Given a text file and search string
#Returns the line that matches the string.
def find_exact_match(filename, search_string):
    with open(filename, 'r') as file:
        for line in file:
            if search_string.lower() in line.strip().lower():
                return line.strip()
    print(f"Could not find {search_string}")
    return None

N_ICL = 8 
N_AVG = 8
N_TRIALS = 4
VIEW_SAMPLES = 5

#Given a start layer and relation
#Creates a dict containing start layer, W, b, and the samples used.
def weight_and_bias(start: int, relation: str) -> dict:
    wdir = f"{start}_27_approx/{relation}"
    ignored_names = ['.ipynb_checkpoints']
    try:
        wnames = os.listdir(wdir)
    except:
        print(f'[weight_and_bias] {wdir} does not exist, skipping')
        return {'i': 0}
        
    j_samples = [x for x in wnames if x not in ignored_names]
    if len(j_samples) < N_AVG:
        print(f'Warning: j_samples has only {len(j_samples)} samples')
    else:
        j_samples = random.sample(j_samples, N_AVG)
    weight_name = f's_o_weight_{start}_27'
    bias_name = f's_o_bias_{start}_27'
    # weight = torch.zeros(4096,4096) #if you want to run just the w2v portion
    # bias = torch.zeros(4096)
    weight = build.mean_weight_or_bias(wdir,weight_name, j_samples).half().to(device)
    bias = build.mean_weight_or_bias(wdir,bias_name, j_samples).half().to(device)
    return {'i': start,
            'weight': weight,
            'bias': bias,
            'j_samples': j_samples}

#Given a start layer and a relation
#Creates a dict containing start layer, offset
def make_offset_dict(start: int, relation: str) -> dict:
    wdir = f"{start}_27_approx_W2V/{relation}"
    ignored_names = ['.ipynb_checkpoints']
    try:
        _ = os.listdir(wdir)
    except:
        print(f'[weight_and_bias] {wdir} does not exist, skipping')
        return {'i': 0}
    offset = torch.load(f'{wdir}/offset_{start}_{end}.pt')
    
    return {'i': start,
            'offset': offset}

#Get dict with dict[i] = i
def get_dict(dicts,i):
    return next((item for item in dicts if item['i'] == i), None)

for beta in [7]:  
    for i in range(0, N_TRIALS):
        for relation in relations:
            json_path = find_exact_match("json_paths.txt", relation)
            file = open(json_path, 'r')
            data = json.load(file)
            file.close()
            
            relation = Relation.from_dict(data)
            prompt_template = relation.prompt_templates[0]
            
            wb_dicts = [weight_and_bias(i, relation.name) for i in range(3,10)]
            # offsets = [make_offset_dict(i, relation.name) for i in range(3,10)]
            
            #We want to show that the bias is necessary and sufficient for conveying morphology
            #To show sufficient:            top_no_bias_correct / lm_at_top_no_bias_correct should be high 
            #To show necessary:             top_no_weight_correct / lm_at_top_no_bias_correct should be low

            top_reg_correct = 0
            top_no_bias_correct = 0
            top_no_weight_correct = 0
            top_w2v_correct = 0
            
            lm_at_top_reg_correct = 0
            lm_at_top_no_bias_correct = 0
            lm_at_top_no_weight_correct = 0
            lm_at_top_w2v_correct = 0
            
            for start in range(3,10):
                clozed_prompts = []
                clozed_answers = []
                
                wb_dict = get_dict(wb_dicts, start)
                #offset_dict = get_dict(offsets, start)
                
                if wb_dict is None:
                    continue
                    
                j_samples = wb_dict['j_samples']
                weight,bias = wb_dict['weight'], wb_dict['bias']
                #offset = offset_dict['offset']
                
                #Do not use the training examples for ICL or testing purposes.
                test_samples = [x for x in relation.samples if x.subject not in j_samples]
                for x in test_samples:
                    test_samples_no_x = [t for t in test_samples if t != x]
                    samples = [x] + random.sample(test_samples_no_x, N_ICL - 1)
                    cloze_prompt = functional.make_prompt(
                        template = prompt_template, 
                        target = x,
                        examples = samples
                        )
                    clozed_prompts.append(cloze_prompt)
                    clozed_answers.append(x.object)
                outputs_lm = functional.predict_next_token(mt=mt, prompt=clozed_prompts)
                preds_lm =  [[x.token for x in xs] for xs in outputs_lm]
                
                reg_correct = 0
                no_bias_correct = 0
                no_weight_correct = 0
                w2v_correct = 0
                lm_correct = 0
                
                for i, sample, objs, prompt, preds in \
                zip(range(0,50), test_samples, clozed_answers, clozed_prompts, preds_lm):
                    if (metrics.any_is_nontrivial_prefix(predictions=[preds[0]], targets=objs)):
                        
                        reg_hs = build.get_hidden_state(mt, prompt, sample.subject, start)
                        
                        #use BWs + b
                        reg_object_hs = reg_hs.mm(weight.t()) * beta + bias
                        reg_preds = build.get_object(mt, reg_object_hs)[0]
            
                        #use Ws
                        no_bias_hs = reg_hs.mm(weight.t())
                        no_bias_preds = build.get_object(mt, no_bias_hs)[0]
    
                        #use s + b
                        no_weight_hs = beta * reg_hs + bias
                        no_weight_preds = build.get_object(mt, no_weight_hs)[0]

                        #use W2v
                        # w2v_hs = reg_hs + offset
                        # w2v_preds = build.get_object(mt, w2v_hs)[0]
                        
                        reg_preds = [x.strip() for x in reg_preds]
                        no_bias_preds = [x.strip() for x in no_bias_preds]
                        no_weight_preds = [x.strip() for x in no_weight_preds]
                        # w2v_preds = [x.strip() for x in w2v_preds]
                        
                        if(metrics.any_is_nontrivial_prefix(predictions=[reg_preds[0]], targets=objs)):
                            reg_correct += 1
            
                        if(metrics.any_is_nontrivial_prefix(predictions=[no_bias_preds[0]], targets=objs)):
                            no_bias_correct += 1
                    
                        if(metrics.any_is_nontrivial_prefix(predictions=[no_weight_preds[0]], targets=objs)):
                            no_weight_correct += 1

                        # if(metrics.any_is_nontrivial_prefix(predictions=[w2v_preds[0]], targets=objs)):
                        #     w2v_correct += 1
                        
                        lm_correct += 1
                        
                if reg_correct > top_reg_correct:
                    top_reg_correct = reg_correct
                    lm_at_top_reg_correct = lm_correct
                    
                if no_bias_correct > top_no_bias_correct:
                    top_no_bias_correct = no_bias_correct
                    lm_at_top_no_bias_correct = lm_correct
                    
                if no_weight_correct > top_no_weight_correct:
                    top_no_weight_correct = no_weight_correct
                    lm_at_top_no_weight_correct = lm_correct
                    
                if w2v_correct > top_w2v_correct:
                    top_w2v_correct = w2v_correct
                    lm_at_top_w2v_correct = lm_correct

            logger.info(f'{relation.name},{beta},{top_reg_correct},{lm_at_top_reg_correct},{top_no_bias_correct},{lm_at_top_no_bias_correct},{top_no_weight_correct},{lm_at_top_no_weight_correct},{top_w2v_correct},{lm_at_top_w2v_correct}')
            