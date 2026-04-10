import argparse
import json
import re
from typing import List, Dict

import evaluate

from client import LLMClient
from open_instruct.eval.gsm.examplars import EXAMPLARS
from prompt_segmentation import SaTSegmentation

def format_examplars(examplars: List[Dict], n_shot = 8) -> str:
    instruction = ''
    for i, exp in enumerate(examplars):
        if i > n_shot: break
        question = exp['question']
        answer = exp['cot_answer']
        instruction += f'Question: {question}\n\nAnswer: {answer}\n\n'
    return instruction

def load_gsm8k(path: str) -> List[Dict]:
    dataset = []
    with open(path, 'r') as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--dataset', type=str, help='path to the dataset')
    parser.add_argument('--base_url', type=str, default='https://0.0.0.0:8000/v1', help='base url of the API')
    parser.add_argument('--api_key', type=str, default='', help='API key for authentication')
    parser.add_argument('--sat_model', type=str, default='sat-12l-sm', help='model for SaT segmentation')
    parser.add_argument('--pattern', type=str, default='<seg>', help='pattern for SaT segmentation')
    parser.add_argument('--n_shot', type=int, default=8, help='number of examplars for few-shot prompting')
    args = parser.parse_args()

    instruction = format_examplars(EXAMPLARS, args.n_shot)
    dataset = load_gsm8k(args.dataset)
    sat = SaTSegmentation(model=args.sat_model, base_url=args.base_url, api_key=args.api_key)
    client = LLMClient(
        base_url=args.base_url, 
        api_key=args.api_key, 
        model=args.model,
    )

    inputs = []
    ground_truths = []
    for data in dataset:
        question = data['question']
        answer = data['answer'].split('####')[-1].strip()
        inputs.append(f'{instruction}Question: {question}\n\nAnswer: ')
        ground_truths.append(answer)
    
    inputs = sat.text_segmentation(inputs, pattern=args.pattern)
    generation_args = {
        'temperature': 0,
        'max_tokens': 512,
        'stop': ['\n\n'],
    }
    outputs = client.generate(inputs, **generation_args)
    preds = []
    for output in outputs:
        output = output.split('####')[-1].strip()
        output = re.sub(args.pattern, '', output)
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            preds.append(numbers[-1])
        else:
            preds.append(output)
    
    exact_match = evaluate.load("hf_exact_match.py")
    em_score = exact_match.compute(references=ground_truths, predictions=preds)
    print(f'Exact Match: {em_score["exact_match"]:.2f}')