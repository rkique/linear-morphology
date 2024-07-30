import torch
import sys
sys.path.append('..')
from transformers import GPTJForCausalLM, AutoTokenizer
import lre.models as models
import lre.functional as functional
import os

import json
import random
from lre.data import Relation, RelationSample, Sequence
import lre.metrics as metrics
import lre.functional as functional
import llra.jacobian as jacobian

#No ICL to allow for the stupid jacobian extraction in llra.jacobian
DEFAULT_N_ICL = 8
device = "cuda:0"
weights = []
biases = []
subjects = []
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
mt = models.ModelAndTokenizer(model,tokenizer)

json_path = 'json/enckno/E06 [animal - youth].json'

with open(json_path, 'r') as file:
    data = json.load(file)
    relation = Relation.from_dict(data)
    prompt_template = relation.prompt_templates[0]
    samples = [[x] + random.sample(relation.samples, DEFAULT_N_ICL - 1) for x in relation.samples]
    prompts = [functional.make_prompt(template=prompt_template, 
                                      target=x, examples=sample) for (x,sample) in zip(relation.samples, samples)]
    # answers = [x.object for x in relation.samples]
    
    for prompt, sample in zip(prompts, relation.samples):
        for i in range(0, 27):
            jacobian.get_jacobians(mt, relation.name, prompt, sample.subject, i)
