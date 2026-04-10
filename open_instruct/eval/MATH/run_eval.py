import argparse
import json
import os
import random
import torch
import vllm
from tqdm import tqdm
from qwen_eval.math_utils import compare_ans

from eval.utils import (
    generate_completions,
    load_hf_lm,
    query_openai_chat_model,
    dynamic_import_function,
    load_hf_tokenizer,
    upload_results_to_hf,
    check_and_upload_model_metadata
)
from eval.MATH.examplars import EXAMPLARS as MATH_EXAMPLARS
from eval.MATH.utilities import last_boxed_only_string, remove_boxed
from eval.MATH.minerva_utils import normalize_final_answer, get_unnormalized_answer, is_equiv

# DEFAULT_PROMPT_PREFIX_COT = "Solve the question below by reasoning step by step, and put the final answer within \\boxed{}."
DEFAULT_PROMPT_PREFIX_COT = "Answer the following question."
DEFAULT_PROMPT_PREFIX_NO_COT = "Answer the following question."

DEFAULT_PROMPT_TEMPLATE_COT = """Question: %s\nAnswer: %s"""
DEFAULT_PROMPT_TEMPLATE_NO_COT = """Question: %s\nAnswer: %s"""

def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["problem"],
                "answer": example["answer"],
                "type": example["subject"]
            })

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)    

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    global MATH_EXAMPLARS
    if args.n_shot:
        if len(MATH_EXAMPLARS) > args.n_shot:
            MATH_EXAMPLARS = random.sample(MATH_EXAMPLARS, args.n_shot)
        demonstrations = []
        for example in MATH_EXAMPLARS:
            if args.no_cot:
                demonstrations.append(
                    ("Question:\n" + example["question"] + "\n\n" + "Answer:",  example["short_answer"])
                )
            else:
                demonstrations.append(
                    "Question: " + example[0] + "\nAnswer: " + example[1]
                )
        initial_demonstrations = args.prompt_prefix + '\n\n\n' + "\n\n\n".join(demonstrations)
    else:
        demonstrations = []

    if args.use_chat_format:
        chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
        def apply_chat_format(example, demonstrations, tokenizer):
            messages = []
            for user_turn, assistant_turn in demonstrations:
                messages.append({"role": "user", "content": user_turn})
                messages.append({"role": "assistant", "content": assistant_turn})
            messages += [{"role": "user", "content":  "Question: " + example["question"].strip() + "\nAnswer: "}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            return prompt
        
    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
                tokenizer_revision=args.hf_revision,
                revision=args.hf_revision,
                gpu_memory_utilization=0.95
            )
            stop_strings = args.additional_stop_sequence + ["Question:"] + ['<|user|>']
            # we only use stop token for non-chat format (usually applied to vanilla pretrained language models).
            # For chat format, we will rely on the model knows when to stop.
            # if not args.use_chat_format:
            #     stop_strings += ["\n"]
            sampling_params = vllm.SamplingParams(
                temperature=0,
                max_tokens=args.max_new_tokens,
                stop=stop_strings, 
            )
            if args.use_chat_format:
                prompts = [apply_chat_format(example, demonstrations, tokenizer) for example in test_data]
            else:
                if args.no_cot:
                    prompts = [initial_demonstrations + "\n\nQuestion:\n" + example["question"].strip() + "\n\nAnswer:\n" for example in test_data]
                else:
                    if args.pause:
                        prompts = [initial_demonstrations + \
    "\n\n\nQuestion: " + example["question"].strip() + "\n" +"<seg>" * 10 + "Answer: Let's think step by step\n" for example in test_data]
                    else:
                        prompts = [initial_demonstrations + "\n\n\nQuestion: " + example["question"].strip() + "\nAnswer: Let's think step by step\n" for example in test_data]
            
                
            if args.seg != -1:
                segmentater = ['<seg>', 'seg', '114', '.$?', 'and', '<and>', '####']
                segmentater = segmentater[args.seg]
                from wtpsplit import SaT
                sat = SaT('sat-12l-sm')
                sat.half().to('cuda:0')
                new_prompts = []
                input_prompts = prompts
                segments = sat.split(input_prompts)
                for seg in tqdm(segments, total=len(input_prompts)):
                    new_prompt = ''
                    for s in seg:
                        if s == '':
                            new_prompt += '\n'
                        else:
                            new_prompt += s.strip() + ' ' + segmentater
                    new_prompts.append(new_prompt)
                input_prompts = new_prompts
                del sat
                prompts = input_prompts
            print(prompts[0])
            

            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
        else:
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path,
                revision=args.hf_revision,
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
            if args.use_chat_format:
                prompts = [apply_chat_format(example, demonstrations, tokenizer) for example in test_data]
            else:
                if args.no_cot:
                    prompts = [initial_demonstrations + "Question:\n" + example["question"].strip() + "\nAnswer:\n" for example in test_data]
                else:
                    prompts = [initial_demonstrations + "Question: " + example["question"].strip() + "\nAnswer:Let's think step by step\n" for example in test_data]
            # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
            stop_tokens = [[tokenizer.encode(stop_seq, add_special_tokens=False)[-1]] for stop_seq in args.additional_stop_sequence]
            # if not args.use_chat_format:
            #     new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            #     stop_tokens += [[new_line_token]]
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=stop_tokens,
                do_sample=False,
            )
    else:
        prompts = [initial_demonstrations + "Question: " + example["question"].strip() + "\nAnswer:" for example in test_data]
        if False:
            from wtpsplit import SaT
            sat = SaT('sat-12l-sm')
            sat.half().to('cuda:0')
            new_prompts = []
            input_prompts = prompts
            segments = sat.split(input_prompts)
            for seg in tqdm(segments, total=len(input_prompts)):
                new_prompt = ''
                for s in seg:
                    if s == '':
                        new_prompt += '\n'
                    else:
                        new_prompt += s.strip() + '<seg>'
                new_prompts.append(new_prompt)
            input_prompts = new_prompts
            del sat
            prompts = input_prompts
        instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, f"openai_math_results.jsonl"),
            max_completion_tokens=768,
            stop=['\n\nQuestion:']

        )
        outputs = [result["output"] for result in results]

    predictions = []
    for output in outputs:
        output = get_unnormalized_answer(output)
        # predictions.append(normalize_final_answer(output))
        predictions.append(output)

    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, outputs, predictions)]

    print("Calculating accuracy...")
    correct_list = []
    for pred in predictions:
        # correct = 1 if is_equiv(pred['prediction'], pred['answer']) else 0
        correct = 1 if compare_ans(pred['prediction'], pred['answer']) else 0
        correct_list.append(correct)
    accuracy = round(sum(correct_list) / len(correct_list), ndigits=4)
    print(f"Accuracy: {accuracy}")
    metrics = {
        "accuracy": accuracy
    }

    # calculate per-type accuracy
    type_correct = {}
    type_total = {}
    for pred, sample in zip(predictions, test_data):
        type_ = sample["type"]
        if type_ not in type_correct:
            type_correct[type_] = 0
            type_total[type_] = 0
        type_correct[type_] += 1 if is_equiv(pred["prediction"], pred["answer"]) else 0
        type_total[type_] += 1
    type_accuracy = {type_: round(type_correct[type_] / type_total[type_], ndigits=4) for type_ in type_correct}
    print("Per-type accuracy:")
    for type_, acc in type_accuracy.items():
        print(f"{type_}: {acc}")
    metrics["per_type_accuracy"] = type_accuracy

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the accuracy
        results = metrics
        task_name = "oi_MATH_cot"
        primary_score = results["accuracy"]
        upload_results_to_hf(
            results,
            args.upload_to_hf,
            args.hf_upload_name,
            task_name=task_name,
            primary_score=primary_score,
            prepend_timestamp=True,
        )
        check_and_upload_model_metadata(
            args.model_name_or_path, args.upload_to_hf, args.hf_upload_name, hf_revision=args.hf_revision
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/gsm"
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/math"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="if specified, we will load the model from a revision of the model in the hub"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--n_shot", 
        type=int, 
        default=8, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="If given, we're evaluating a model without chain-of-thought."
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1024,
        help="maximum number of tokens to generate for each prompt."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default=None,
        help="the specific prefix to use for instructing the model."
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="the specific template to use for instructing the model."
    )
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
    )
    parser.add_argument(
        "--upload_to_hf",
        type=str,
        default=None,
        help="If specified, we will upload the results to Hugging Face Datasets. "
             "This should be the name of the dataset to upload to."
    )
    parser.add_argument(
        "--hf_upload_name",
        type=str,
        default=None,
        help="If uploading to hf, this is the model name"
    )
    parser.add_argument(
        "--seg",
        type=int,
        default=-1,
        help="If specified, we will split the input prompt into segments"
    )
    parser.add_argument(
        '--pause',
        action='store_true'
    )
    args = parser.parse_args()

    # update the prompt prefix depending on whether CoT is being used
    if args.prompt_prefix is None:
        args.prompt_prefix = DEFAULT_PROMPT_PREFIX_NO_COT if args.no_cot else DEFAULT_PROMPT_PREFIX_COT

    # update the prompt template depending on whether CoT is being used
    if args.prompt_template is None:
        args.prompt_template = DEFAULT_PROMPT_TEMPLATE_NO_COT if args.no_cot else DEFAULT_PROMPT_TEMPLATE_COT

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)