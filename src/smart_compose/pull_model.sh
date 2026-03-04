#!/bin/bash

echo "Pulling distilgpt2 files from Hugging Face..." && sleep 1
mkdir -p ./distilgpt2_local

curl -L "https://huggingface.co/distilgpt2/resolve/main/config.json" \
	-o ./distilgpt2_local/config.json

curl -L "https://huggingface.co/distilgpt2/resolve/main/tokenizer.json" \
	-o ./distilgpt2_local/tokenizer.json

curl -L "https://huggingface.co/distilgpt2/resolve/main/tokenizer_config.json" \
	-o ./distilgpt2_local/tokenizer_config.json

curl -L "https://huggingface.co/distilgpt2/resolve/main/vocab.json" \
	-o ./distilgpt2_local/vocab.json

curl -L "https://huggingface.co/distilgpt2/resolve/main/merges.txt" \
	-o ./distilgpt2_local/merges.txt

curl -L "https://huggingface.co/distilgpt2/resolve/main/model.safetensors" \
	-o ./distilgpt2_local/model.safetensors

echo "✅ Done! Files saved to ./distilgpt2_local"
