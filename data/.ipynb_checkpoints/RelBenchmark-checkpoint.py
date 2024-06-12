import sys
import json
sys.path.append('..')
from lre.data import Relation, RelationSample, Sequence
from lre.operators import JacobianIclMeanEstimator, Word2VecIclEstimator
import lre.functional as functional
import lre.models as models
import lre.metrics as metrics
import lre.logging_utils as logging_utils
from collections import defaultdict
from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import random
import time
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='word2vec_results.txt',
    level=logging.INFO,
    format = logging_utils.DEFAULT_FORMAT,
    datefmt=logging_utils.DEFAULT_DATEFMT,
)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

counts_by_lre_correct: dict[bool, int] = defaultdict(int)

def test_operator_on_relation(operator, relation, mt, h_layer, z_layer):
    logger.info(f'starting test: {operator} on {relation}')
    prompt_template = relation.prompt_templates[0]
    clozed_prompts = []
    clozed_answers = []
    #For each sample...
    for x in relation.samples:
        cloze_prompt = functional.make_prompt(
            template = prompt_template, 
            target = x,
            examples = relation.samples
            )
        clozed_prompts.append(cloze_prompt)
        clozed_answers.append(x.object)

    #LM PREDICTION
    #start_time = time.time()
    outputs_lm = functional.predict_next_token(mt=mt, prompt=clozed_prompts)
    preds_lm =  [[x.token for x in xs] for xs in outputs_lm]
    recall_lm = metrics.recall(preds_lm, clozed_answers)
    #end_time = time.time()
    #logging.info(f'total LM prediction time: {end_time - start_time} seconds with recall {recall_lm}')

    #OPERATOR PREDICTION
    start_time = time.time()
    logging.info(f'starting operator prediction')
    operator = operator(mt=mt, h_layer=h_layer, z_layer=z_layer)
    operator = operator(relation)
    end_time = time.time()

    logging.info(f'total operator prediction time: {end_time - start_time} seconds')

    outputs_lre = []
    for sample in relation.samples:
        output_lre = operator(sample)
        outputs_lre.append(output_lre.predictions)

    #remember that predictions is made up of (token,probs)
    preds_lre = [[x.token for x in xs] for xs in outputs_lre]
    recall_lre = metrics.recall(preds_lre, clozed_answers)

    preds_by_lre_correct = defaultdict(list)
    targets_by_lre_correct = defaultdict(list)
    
    for pred_lm, pred_lre, target in zip(preds_lm, preds_lre, clozed_answers):
        lm_correct = metrics.any_is_nontrivial_prefix(pred_lm, target)
        if lm_correct:
          lre_correct = metrics.any_is_nontrivial_prefix(pred_lre, target)
          logging.info(f'{pred_lre} matches {target} is {lre_correct}')
          preds_by_lre_correct[lre_correct].append(pred_lre)
          targets_by_lre_correct[lre_correct].append(target)
          counts_by_lre_correct[lre_correct] += 1
          #logging.info(f'{pred_lre},{target}')

    logging.info(f'{relation.name} ({len(relation.samples)}) total: {counts_by_lre_correct}')
import os

def all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            file_paths.append(relative_path)
    return file_paths
directory = 'json'
file_paths = all_file_paths('json')

device = "cuda"

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
mt = models.ModelAndTokenizer(model,tokenizer)
logging.info('model + tokenizer loaded')

def test_operator_on_json(operator, json_path, mt, h_layer, z_layer):
    with open(json_path, 'r') as file:
        data = json.load(file)
        relation = Relation.from_dict(data)
        assert all(isinstance(sample, RelationSample) for sample in relation.samples)
        test_operator_on_relation(operator, relation, mt, 5, 27)

json_path = 'json/enckno/E06 [animal - youth].json'
#test_operator_on_json(Word2VecIclEstimator, json_path, mt, 5, 27)
test_operator_on_json(JacobianIclMeanEstimator, json_path, mt, 5, 27)

#for json_path in file_paths:
#    json_path = 'json/' + json_path
    #test_operator_on_json(Word2VecIclEstimator, json_path, mt, 5, 27)
    #test_operator_on_json(JacobianIclMeanEstimator, json_path, mt, 5, 27)
