import os
import sys
import json
import random
sys.path.append('..')
import logging

import lre.models as models
from lre.data import Relation, RelationSample, Sequence
import lre.functional as functional
import lre.metrics as metrics
import lre.logging_utils as logging_utils

from transformers import GPTJForCausalLM, AutoTokenizer
import numpy as np
import torch
import llra.viz as viz
import llra.build as build
import lre.functional as functional
import numpy as np
from collections import defaultdict

device = 'cuda:0'
logger = logging.getLogger(__name__)

#MAKE SURE TO CHANGE THIS EACH RUN
RESULTS_FILE = 'results/Wednesday_Cuda1_TestOrientedLRE.txt'

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
model.to(device)

build.determine_device(mt)
build.determine_params(mt)


#OBTAIN LRE WEIGHTS: Method 1
#Load weights saved with new naming convention
ignored_relations = ['LORE hypernyms - animals', '.ipynb_checkpoints']
relations = [x for x in os.listdir('7_27_approx') if x not in ignored_relations]

start, end = 7, 27
APPROX_FOLDER = f'{start}_{end}_approx'
WEIGHT_NAME = f's_o_weight_{start}_{end}'
BIAS_NAME = f's_o_bias_{start}_{end}'

DEFAULT_N_ICL = 8
N_TRIALS = 8
VIEW_SAMPLES = 3

#
#samples = ["student", "spouse", "singer", "parishioner", "employee", "elephant", "citizen", "bird"]
#print(samples)

relations = [
 # 'name - nationality',
 # 'animal - youth',
 # 'verb_Ving - 3pSg',
 # 'noun+less_reg',
 # 'verb+able_reg',
 # 'UK_city - county',
 # 'antonyms - binary',
 # 'verb_inf - 3pSg',
 # 're+verb_reg',
 # 'verb_inf - Ved',
 # 'country - language',
 # 'meronyms - part',
 # 'verb_Ving - Ved',
 # 'animal - shelter',
 # 'hypernyms - misc',
 # 'meronyms - substance',
 # 'noun - plural_irreg',
 # 'un+adj_reg',
 # 'verb+ment_irreg',
 # 'adj+ness_reg',
 # 'over+adj_reg',
 'verb+er_irreg',
 # 'adj+ly_reg',
 # 'name - occupation',
 # 'synonyms - intensity',
 # 'animal - sound',
 # 'noun - plural_reg',
 # 'Ving - verb_inf',
 # 'male - female',
 # 'verb_3pSg - Ved',
 # 'meronyms - member',
 # 'things - color',
 # 'hyponyms - misc',
 # 'adj - superlative',
 # 'verb+tion_irreg',
 # 'synonyms - exact',
 # 'hypernyms - animals',
 # 'country - capital'
]

def find_exact_match(filename, search_string):
    with open(filename, 'r') as file:
        for line in file:
            if search_string.lower() in line.strip().lower():
                return line.strip()
    print(f"Could not find {search_string}")
    return None

betas = [1,2,4,6,8,10,12,14]
for relation_name in relations:
    print(f'testing {relation_name}')
    for _ in range(0,4):
        for beta in betas:
            #LOAD RANDOM SAMPLES
            wdir = f"{APPROX_FOLDER}/{relation_name}"
            ignored_names = ['.ipynb_checkpoints', 'OLD']
            samples = [x for x in os.listdir(wdir) if x not in ignored_names]
            #print(f'len of samples is {len(samples)}')
            samples = random.sample(samples,DEFAULT_N_ICL)
            #samples = ['intrude', 'provide', 'publish', 'bake', 'destroy', 'deliver', 'entertain', 'compose']
            reg_weight = build.mean_weight_or_bias(wdir,
                                                   WEIGHT_NAME, samples).half().to(device)
            reg_bias = build.mean_weight_or_bias(wdir,
                                                 BIAS_NAME, samples).half().to(device)
            
            #LOAD SPACED SAMPLES
            wdir = f"spaced_er_6_27_approx/{relation_name}"
            ignored_names = ['.ipynb_checkpoints', 'OLD']
            samples = [x for x in os.listdir(wdir) if x not in ignored_names]
            spaced_weight = build.mean_weight_or_bias(wdir,
                                                   WEIGHT_NAME, samples).half().to(device)
            spaced_bias = build.mean_weight_or_bias(wdir,
                                                 BIAS_NAME, samples).half().to(device)

            #LOAD RELATION FOR TESTING
            json_path = find_exact_match("json_paths.txt", relation_name)
            file = open(json_path, 'r')
            data = json.load(file)
            file.close()
            relation = Relation.from_dict(data)
            prompt_template = relation.prompt_templates[0]
            
            #ASSEMBLE PROMPTS AND OBJECT ANSWERS
            clozed_prompts = []
            clozed_answers = []
            random.shuffle(relation.samples)
            for x in relation.samples:
                samples = [x] + random.sample(relation.samples, DEFAULT_N_ICL - 1)
                #print(f'{samples} samples)')
                cloze_prompt = functional.make_prompt(
                    template = prompt_template, 
                    target = x,
                    examples = samples
                    )
                clozed_prompts.append(cloze_prompt)
                clozed_answers.append(x.object)
            
            outputs_lm = functional.predict_next_token(mt=mt, prompt=clozed_prompts)
            preds_lm =  [[x.token for x in xs] for xs in outputs_lm]
            recall_lm = metrics.recall(preds_lm, clozed_answers)
            
            reg_correct = 0
            spaced_correct = 0
            lm_correct = 0
            
            for i, sample, objs, prompt, preds in \
            zip(range(0,50), relation.samples, clozed_answers, clozed_prompts, preds_lm):
                
                if (metrics.any_is_nontrivial_prefix(predictions=[preds[0]], targets=objs)):
                    #uses the regular LRE
                    reg_hs = build.get_hidden_state(mt, prompt, sample.subject, start)
                    reg_hs_end = build.get_hidden_state(mt, prompt, sample.subject, end)
                    reg_object_hs = reg_hs.mm(reg_weight.t()) * beta + reg_bias
                    reg_preds = build.get_object(mt, reg_object_hs)[0]

                    # spaced_object_hs = reg_hs.mm(spaced_weight.t()) * beta + spaced_bias
                    # spaced_preds = build.get_object(mt, spaced_object_hs)[0]
        
                    if(metrics.any_is_nontrivial_prefix(predictions=[reg_preds[0]], targets=objs)):
                        reg_correct += 1
                            
                    # if(metrics.any_is_nontrivial_prefix(predictions=[spaced_preds[0]], targets=objs)):
                    #     spaced_correct += 1
                    
                    if(i < VIEW_SAMPLES):
                        pass
                    lm_correct += 1
                
            #print(f'beta, name, Ws+b, Ws')
            logger.info(f'{beta},{relation.name},{reg_correct},{spaced_correct},{lm_correct}')         