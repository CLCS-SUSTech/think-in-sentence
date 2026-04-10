## Environment

This project utilizes code from an early version of the `open_instruct` repository. Since `open_instruct` does not conduct version control in their early version, the exact source of the code used in this project is not traceable.

If you need to perform fine-tuning, we recommend writing your own Hugging Face Trainer script based on the methods described in the paper, or customizing your workflow using repositories like `llama-factory`.

Installation Steps:
```shell
# Navigate to the directory:
cd open_instruct

# Install the necessary dependencies:
pip install -r requirements.txt

# Install the specific version of wtpsplit:
pip install wtpsplit==2.1.4
```

## Example

We prepare a simple example to conduct sentence-by-sentence inference using GSM8k dataset.

Launch an OpenAI compatible service using any inference engine.
```shell
vllm serve Qwen/Qwen2.5-7B --served-model-name qwen2.5-7b --port 8000
```

Run `example_gsm8k.py`.
```shell
python -m example_gsm8k --model qwen2.5-7b --dataset data/gsm8k
```