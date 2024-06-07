import sys
sys.path.append('..')
from lre.data import Relation, RelationSample
from lre.operators import JacobianIclEstimator, Word2VecIclEstimator
import lre.functional as functional
import lre.metrics as metrics
import lre.models as models
import torch
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"running A2Rel with {device}")
mt = models.load_model("gptj", device=device, fp16=True)

def relation_from_path(path, relation_name, prompt):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.replace('\n','') for line in lines]

        for line in lines:
            a, bs = line.split('\t')
            subjects.append(a)
            bs = bs.split('/')
            subject_object_pairs.append((a,bs[0]))
            for b in bs:
                all_pairs.append((a,b))

    pairs = subject_object_pairs
    RelationSamples = [RelationSample(*pair) for pair in pairs]
    rel = Relation(
                    name=relation_name,
                    prompt_templates=[prompt],
                    prompt_templates_zs=[],
                    samples=
                    RelationSamples
                )
    return rel

def test_operator_on_relation(operator, relation, mt, h_layer, z_layer, k):
    operator = operator(mt=mt, h_layer=h_layer, z_layer=z_layer)
    operator = operator(relation)
    prompt_template = relation.prompt_templates[0]

    #assemble in-context prompts
    clozed_prompts = []
    clozed_answers = []
    for x in relation.samples:
        clozed_samples = [s for s in relation.samples if s != x]
        cloze_template = functional.make_prompt(
            prompt_template=prompt_template,
            subject="{}",
            examples = clozed_samples
            )
        cloze_prompt = cloze_template.format(x.subject)
        clozed_prompts.append(cloze_prompt)
        clozed_answers.append(x.object)

    for prompt in (clozed_prompts):
        print(f'Prompt: \n{prompt}\n')

    #LM prediction. max-tokens: 2048
    outputs_lm = functional.predict_next_token(mt=mt, prompt=clozed_prompts, k=k)
    preds_lm =  [[x.token for x in xs] for xs in outputs_lm]
    recall_lm = metrics.recall(preds_lm, clozed_answers)

    outputs_lre = []
    for sample in relation.samples:
        print(f'operator has {type(operator)} (should be LinearRelationOperator)')
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

    print(f'For {type(operator)} on {relation.name} (out of correct): {counts_by_lm_correct}')

#subj. rep layer: 5
#output layer: 27 (last)

for i in range(0,3):
    subjects = []
    subject_object_pairs = []
    all_pairs = []
    animal_path = f'data/enckno/animal-young{i}.txt'
    animal_rel = relation_from_path(animal_path, f"animal-youth{i}", "The young version of {} is")
    counts_by_lm_correct: dict[bool, int] = defaultdict(int)
    test_operator_on_relation(JacobianIclEstimator, animal_rel, mt, 5, 27,5)
    test_operator_on_relation(Word2VecIclEstimator, animal_rel, mt, 5, 27,5)