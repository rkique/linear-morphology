#Given a relation dataset, this script filters the relation samples to those which are known by the LM. 

import json
from lre.functional import filter_relation_samples
from lre.data import Relation, RelationSample
import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import lre.models as models
import lre.logging_utils as logging_utils
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='word2vec_results.txt',
    level=logging.INFO,
    format = logging_utils.DEFAULT_FORMAT,
    datefmt=logging_utils.DEFAULT_DATEFMT,
)

logger.addHandler(logging.StreamHandler())

GPT_J_PATH = "EleutherAI/gpt-j-6B"
device = 'cuda'


kwargs = {'revision': "float16", 
          'torch_dtype': torch.float16,
          'low_cpu_mem_usage': True
         }

model = GPTJForCausalLM.from_pretrained(GPT_J_PATH, **kwargs)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(GPT_J_PATH)
tokenizer.pad_token = tokenizer.eos_token
mt = models.ModelAndTokenizer(model,tokenizer)

rel_path = 'json/enckno/E06 [animal - youth]'
json_path = f'{rel_path}.json'

with open(json_path, 'r') as file:
    data = json.load(file)
    relation = Relation.from_dict(data)
    prompt_template = relation.prompt_templates[0]

    with open(rel_path + '.txt') as file:
        json.dump(relation.samples.to_json(), file, indent=4)

    relation = filter_relation_samples(mt=mt, 
                            test_relation=relation,
                            prompt_template=prompt_template,
                            examples=relation.samples)
    
    with open(rel_path + '_updated.txt') as file:
        json.dump(relation.samples.to_json(), file, indent=4)