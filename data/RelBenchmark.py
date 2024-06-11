import sys
import json
sys.path.append('..')
from lre.data import Relation, RelationSample
from lre.operators import JacobianIclEstimator, Word2VecIclEstimator
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

logger.addHandler(logging.StreamHandler())

counts_by_lm_correct: dict[bool, int] = defaultdict(int)

def test_operator_on_relation(operator, relation, mt, h_layer, z_layer, n_icl=8, k=5):
    logger.info(f'starting test: {operator} on {relation}')
    prompt_template = relation.prompt_templates[0]
    clozed_prompts = []
    clozed_answers = []
    #For each sample...
    for x in relation.samples:
        #make the prompt
        cloze_prompt = functional.make_prompt(
            prompt_template=prompt_template,
            subject=x,
            examples = relation.samples
            )
        # cloze_prompt = cloze_template.format(x.subject)
        clozed_prompts.append(cloze_prompt)
        clozed_answers.append(x.object)

    #LM prediction
    start_time = time.time()
    logging.info(f'starting next token prediction')
    outputs_lm = functional.predict_next_token(mt=mt, prompt=clozed_prompts, k=k)
    preds_lm =  [[x.token for x in xs] for xs in outputs_lm]
    recall_lm = metrics.recall(preds_lm, clozed_answers)
    end_time = time.time()
    logging.info(f'total LM prediction time: {end_time - start_time} seconds with recall {recall_lm}')

    #operator prediction
    start_time = time.time()
    logging.info(f'starting operator prediction')
    operator = operator(mt=mt, h_layer=h_layer, z_layer=z_layer)
    operator = operator(relation)
    end_time = time.time()
    logging.info(f'total operator prediction time: {end_time - start_time} seconds')

    outputs_lre = []
    for sample in relation.samples:
        output_lre = operator(sample.subject, k=k)
        outputs_lre.append(output_lre.predictions)

    #remember that predictions is made up of (token,probs)
    preds_lre = [[x.token for x in xs] for xs in outputs_lre]
    recall_lre = metrics.recall(preds_lre, clozed_answers)

    preds_by_lm_correct = defaultdict(list)
    targets_by_lm_correct = defaultdict(list)

    #if the LM was correct, append pred_lre to preds_by_lm_correct (sth like {True: 5, False: 2})
    for pred_lm, pred_lre, target in zip(preds_lm, preds_lre, clozed_answers):
        lm_correct = metrics.any_is_nontrivial_prefix(pred_lm, target)
        preds_by_lm_correct[lm_correct].append(pred_lre)
        targets_by_lm_correct[lm_correct].append(target)
        counts_by_lm_correct[lm_correct] += 1

    print(f'For {type(operator)} on {relation.name} (out of correct, with {len(relation.samples)} total): {counts_by_lm_correct}')

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

for json_path in file_paths:
    with open('json/' + json_path, 'r') as file:
        data = json.load(file)
        
        relation = Relation.from_dict(data)
        assert all(isinstance(sample, RelationSample) for sample in relation.samples)
        
        logging.info(f'[{relation.name}] Loading GPT-J and tokenizer')
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        logging.info('Model loaded')
        model.to(device)
        logging.info('Model put on cuda')

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        logging.info('Tokenizer loaded')
        tokenizer.pad_token = tokenizer.eos_token

        mt = models.ModelAndTokenizer(model,tokenizer)

        #8 ICL examples, 50 different samples total.
        test_operator_on_relation(Word2VecIclEstimator, relation, mt, 5, 27, k=5)
        #test_operator_on_relation(JacobianIclEstimator, relation, mt, 5, 27, k=5)
