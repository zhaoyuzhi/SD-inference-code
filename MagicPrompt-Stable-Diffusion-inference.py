#!/usr/bin/env python
# coding: utf-8

import os
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

model_path = "E:\\pre-trained models\\huggingface\\models AIGC\\MagicPrompt-Stable-Diffusion"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100)

input_short_description = 'a beautiful woman standing indoor'

out = pipe(input_short_description)
print(out)
print(out[0]['generated_text'])
