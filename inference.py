import argparse 
from typing import Dict, List, Any
import json 
import os 
from tqdm import tqdm 
import bitsandbytes 
from math import ceil 
import re 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch 
from datasets import load_dataset, Dataset 
from functools import partial 
from dotenv import load_dotenv
from collections import defaultdict 
# from UniEval.utils import convert_to_json
# from UniEval.metric.evaluator import get_evaluator

load_dotenv() 

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

MODEL_CONFIGS = {
    'tablellama': {
        'model_id': "osunlp/TableLlama",
        'model_path': "osunlp/TableLlama"
    },
    'llama3': {
        'model_id': "meta-llama/Meta-Llama-3-8B-Instruct",
        'model_path': "/shared/nas2/shared/llms/Meta-Llama-3-8B-Instruct"
    },
    'llama3-text': {
        'model_id': "meta-llama/Meta-Llama-3-8B",
        "model_path": "/shared/nas2/shared/llms/Meta-Llama-3-8B"
    },
    'llama2' : {
        'model_id': '/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-7b-hf',
        'model_path': "/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-7b-hf"
    },
    "mistral": {
        'model_id': "mistralai/Mistral-7B-Instruct-v0.2",
        "model_path": "/shared/nas2/shared/llms/Mistral-7B-Instruct-v0.2"
    }
}

def convert_table_to_llama_format(column_names, content_values):
    # Start the table with the header
    new_table = "|| " + " | ".join(column_names) + " ||\n"
    
    # Process each row of the content values
    for row in content_values:
        # Format each row as a part of the new table format
        new_table += "|| " + " | ".join(row) + " ||\n"
    
    return new_table

def convert_table_to_tl_format(table_column_names, table_content_values):
    # Define the official table format components
    official_format = "[TAB] | "
    official_format += " | ".join(table_column_names) + " |"
    official_format += " [SEP]\n"
    
    # Process each row in the content values
    for row in table_content_values:
        official_format += "| " + " | ".join(row) + " |\n"
        official_format += "[SEP] | "
    
    # Remove the last trailing "[SEP] | " and newline
    official_format = official_format.rstrip("[SEP] | \n")
    return official_format


def construct_context(batch_ins:Dict[str, List], tokenizer, train_strategy, context_type: str='none', prompt='gen', use_template:bool=True) -> Dict[str, List]:
    '''
    
    :use_template: for base models, do not need to use template 
    '''
    new_fields = {
        'text': [],
        'text_len': []
    }
    batch_size = len(batch_ins['paper']) 
    for idx in range(batch_size):
        # question = f"The current time is {batch_ins['question_date'][idx]}." + batch_ins['question_sentence'][idx]
        table_caption = batch_ins['table_caption'][idx]
        table_column_names = batch_ins['table_column_names'][idx]
        table_content_values = batch_ins['table_content_values'][idx]
        claim = batch_ins['claim'][idx]
        label = batch_ins['label'][idx]

        if args.model == 'tablellama':
            table = convert_table_to_tl_format(table_column_names, table_content_values)
        else:
            table = convert_table_to_llama_format(table_column_names, table_content_values)

        
        if args.model == 'tablellama':

            messages = [{"role": "user", "content": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Instructio: This is a table fact verification task. The goal of this task is to distinguish whether the given statement is entailed or refuted by the given table. Input: {table_caption} Table: {table} Question: The statement is <{claim}> Is it entailed or refuted by the table above? Answer \"entailed\", \"refute\", or \"not enough info\"."}]
                        # {"role": "user", "content": f"Which country won the 2023 Eurovision Song Contest? "},
                        # {"role": "assistant", "content": f"The question is related to the following information. The document was published on 2023/05/13. Liverpool, United Kingdom CNN \u2014\n\nSweden\u2019s Loreen has won the Eurovision Song Contest for a second time, earning a historic triumph at an extravagant and crowd-pleasing show held in Liverpool, United Kingdom, on behalf of Ukraine.\n\nShe became just the second performer to win the competition more than once, clinching victory with pop ballad \u201cTattoo\u201d and cementing her legacy at the kitsch and wildly celebrated music contest.\n\nLoreen had previously won the contest in Baku in 2012, with her career-altering hit \u201cEuphoria. The answer for this question is Sweden."},

                        # {"role": "user", "content": f"{question}"}]

        else:
            messages = messages = [{"role": "user", "content": f"Read the following table and then answer a question. Caption: {table_caption} Table: {table} Claim: {claim} Question: Is the claim true or false? Answer \"support\", \"refute\", or \"not enough info\"."}]

            
        # if use_template:
        #     text = tokenizer.apply_chat_template(
        #         messages,
        #         add_generation_prompt=True,
        #         tokenize=False)
        # else:
        text = messages[0]['content'] + "\n<answer>"

        new_fields['text'].append(text)
        new_fields['text_len'].append(len(tokenizer.tokenize(text))) # text_len is not reliable because of padding 

    return new_fields 

def compute_qa_metrics(gold:List[str], predicted: str):

    gold_answers = [re.sub(r'[^\w\s]','', x) for x in gold]
    gold_answers = [x for x in gold_answers if x!='']
    assert len(gold_answers) >0 
    # Tokenize true and predicted answers
    predicted_answer = re.sub(r'[^\w\s]','',predicted)
    predicted_answer = predicted_answer.strip() 
    pred_tokens = set(predicted_answer.lower().split())
    if len(pred_tokens) == 0:
        return {
        'f1': 0.0, 
        'include_em': 0.0,
        'real_em': 0.0,
        'length': 0.0
        }

    best_f1 = 0.0
    best_length_ratio = 1000

    for true_answer in gold_answers:
        answer_tokens = set(true_answer.lower().split()) 
        # Calculate intersection and union of tokens
        length_ratio = len(pred_tokens) * 1.0 / len(answer_tokens)
        common_tokens = answer_tokens  & pred_tokens
        num_common_tokens = len(common_tokens)
        
        prec = 1.0 * num_common_tokens / len(pred_tokens)
        recall = 1.0 * num_common_tokens / len(answer_tokens )
        # Calculate F1 score
        if num_common_tokens == 0:
            f1_score=0.0
        else:
            f1_score = 2 * (prec * recall) / (prec + recall)
            best_f1 = max(f1_score, best_f1)

        if abs(length_ratio - 1.0) < abs(best_length_ratio - 1.0) :
            best_length_ratio = length_ratio 


    include_em, real_em = 0.0, 0.0
    for true_answer in gold_answers:
        if predicted_answer.lower() == true_answer.lower(): real_em = 1.0
        if  true_answer.lower() in predicted_answer.lower(): include_em = 1.0

    consistency=0.0
    # for true_answer in gold_answers:
    #     data = convert_to_json(output_list=[predicted_answer], src_list=[true_answer.lower()])
    #     # Initialize evaluator for a specific task
    #     evaluator = get_evaluator('fact')
    #     # Get factual consistency scores
    #     consistency = evaluator.evaluate(data, print_result=False)[0]

    return {
        'f1': best_f1, 
        'include_em': include_em,
        'real_em': real_em,
        'consistency': consistency,
        'length': best_length_ratio
    }

if __name__ == '__main__':
    os.environ['HF_TOKEN'] = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_batch_size', type=int, default=4) 
    parser.add_argument('--prompt', type=str, default='gen')

    parser.add_argument('--model', type=str, choices=['tablellama', 'llama3','llama2', 'mistral','llama3-text'], default='tablellama')
    args = parser.parse_args() 


    train_strategy = '' # realtime200  realtime200sameSIU_indexqa indexDocSamerealtime200 realtime200Inst_indexqa  realtime200SIU_indexqa indexDocrealtime200  realtime200SIU_indexChunkqa  indexChunkrealtime200  indexSumChunkrealtime200 indexDocSamerealtime200

    # base model pretrain/index
    localpath='tablellama'


    if '50' in train_strategy:
        sample_num = '50'

    dataset = load_dataset("json", data_files="./sci_tab.json", token=os.environ['HF_TOKEN'])['train'] # train is the default split here 

    model = AutoModelForCausalLM.from_pretrained("osunlp/TableLlama")
    tokenizer = AutoTokenizer.from_pretrained("osunlp/TableLlama", token=os.environ['HF_TOKEN'], padding_side='left')
    # The warning is missleading, but you need sentencepiece installed if the tokenizer you are trying to load does not have a tokenizer.json (FYI @itazap )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    # if 'lr' in localpath:
    #     config = AutoConfig.from_pretrained(
    #     localpath
    #     )
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         localpath, access_token=os.environ['HF_TOKEN'], padding_side='left'
    #     )
    #     terminators = [
    #         tokenizer.eos_token_id,
    #         tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #         ]
    #     model = transformers.AutoModelForCausalLM.from_pretrained(
    #         localpath,
    #         config=config,
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #     )
    #     args.model='ckpt'
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[localpath]['model_id'], access_token=os.environ['HF_TOKEN'], padding_side='left')
    #     terminators = [
    #         tokenizer.eos_token_id,
    #         tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #         ]
    
    #     model = AutoModelForCausalLM.from_pretrained(
    #         MODEL_CONFIGS[args.model]['model_path'],
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #     )
    #     # args.model='zeroshot'


    tokenizer.pad_token = tokenizer.eos_token
     
    tokenized_dataset = dataset.map(partial(construct_context, tokenizer=tokenizer, train_strategy=train_strategy, prompt=args.prompt, context_type='none'), batched=True) 
    tokenized_dataset = tokenized_dataset.map(lambda x: tokenizer(x['text'],padding=True, max_length=2048, truncation=True), 
        batched=True,  
        batch_size=args.gen_batch_size ) # batch size must match for truncation + padding 
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    

    result_list = []
    start = 0
    qa_metrics = defaultdict(list)
    qa_metrics = []

    total_batches = ceil(len(tokenized_dataset) /args.gen_batch_size ) 
    total_batches = 25
    for batch_idx in tqdm(range(total_batches)):
        end = min(len(tokenized_dataset), start+ args.gen_batch_size) 
        batch = tokenized_dataset[start: end]
        start += args.gen_batch_size 

        assert(isinstance(batch['input_ids'], torch.Tensor))
        outputs = model.generate(
            input_ids=batch['input_ids'].to(model.device), 
            attention_mask = batch['attention_mask'].to(model.device), 
            max_new_tokens=256,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id, 
            do_sample=False, # greedy decoding 
            temperature=None,
            top_p=None
        )

        for idx, output_ids in enumerate(outputs):
            input_len = batch['input_ids'][idx].shape[-1]
            result = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True) # str 
            # prompt = tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=True) 
            # options = batch['choices'][idx]
            # answer_text = [options[int(x)] for x in batch['answer'][idx]]
            m = re.search(r'<answer>([^<]+)', result)
            if m:
                predicted = m.group(1)

            else:
                predicted = result 

            results = {
                'claim': batch['claim'][idx],
                'label': batch['label'][idx],
                'predicted': predicted, 
                'generated': result 
            }
            predict_label = ''
            try:
                if args.model=='tablellama':
                    if 'refute' in predicted:
                        predict_label = 'refuted'
                    elif 'entail' in predicted:
                        predict_label = 'supported'
                    elif 'not enough' in predicted:
                        predict_label = 'not enough info'
                else:
                    if 'refute' in predicted:
                        predict_label = 'refuted'
                    elif 'support' in predicted:
                        predict_label = 'supported'
                    elif 'not enough' in predicted:
                        predict_label = 'not enough info'
                results.update({'predict_label':predict_label})
                success = 0
                if predict_label == results['label']:
                    success = 1
                    results.update({'success':success})
                else:
                    results.update({'success':success})
                qa_metrics.append(success)
    
            except AssertionError:
                print(f"question {batch['question_id'][idx]} has no gold answer")
                continue 

            result_list.append(results)

    

    filename = f'output/{args.model}.json' 


    with open(filename,'w') as f:
        json.dump(result_list, f, indent=2)

    print('accuracy: '+str(sum(qa_metrics*1.0)/len(qa_metrics)))

    # final_metrics = {} 
    # for metric in qa_metrics:
    #     final_metrics[metric] = sum(qa_metrics[metric]) *1.0 / len(qa_metrics[metric]) 

    

    # with open(f'output/realtimeQA_llama2/base_{train_strategy}_{args.prompt}_metrics.json','w') as f:
    #     json.dump(final_metrics, f, indent=2) 

