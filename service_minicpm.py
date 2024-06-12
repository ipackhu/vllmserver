#!/usr/bin/env python
# encoding: utf-8
from PIL import Image
import traceback
import re
import os
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import json

# README, How to run demo on different devices

# For Nvidia GPUs.
# python web_demo_2.5.py --device cuda

# For Mac with MPS (Apple silicon or AMD GPUs).
# PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo_2.5.py --device mps

# Argparser
# parser = argparse.ArgumentParser(description='demo')
# parser.add_argument('--device', type=str, default='cuda', help='cuda or mps')
# args = parser.parse_args()
device = 'cuda' #args.device
#assert device in ['cuda', 'mps']


model = None
tokenizer = None

def init_model(model_path, lora_weights):
    global model, tokenizer
    # Load model
    #model_path = '.'# 'openbmb/MiniCPM-Llama3-V-2_5'
    #lora_weights = '/v1/big_model/train_vllm/minicpmv/output/output_minicpmv2_lora/checkpoint-100'
    if 'int4' in model_path:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.float16)
        model = model.to(device=device)
        if lora_weights and os.path.exists(lora_weights):
             model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()



ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 2.5'


def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    default_params = {"num_beams":1, "repetition_penalty": 1.2, "max_new_tokens": 1024}
    if params is None:
        params = default_params
    if img is None:
        return -1, "Error, invalid image, please upload a new image", None, None
    try:
        image = img.convert('RGB')
        answer = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        res = re.sub(r'(<box>.*</box>)', '', answer)
        res = res.replace('<ref>', '')
        res = res.replace('</ref>', '')
        res = res.replace('<box>', '')
        answer = res.replace('</box>', '')#.replace('\n', '\\n')
        return -1, answer, None, None
    except Exception as err:
        print(err)
        traceback.print_exc()
        return -1, ERROR_MSG, None, None


def respond(_question, img, repetition_penalty, top_p, top_k, temperature, max_new_token=896):
    _context = [{"role": "user", "content": _question}]
    print('<User>:', _question)

    params = {
        'sampling': True,
        'top_p': top_p,
        'top_k': top_k,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        "max_new_tokens": max_new_token,
        #"stream": True
    }
    code, _answer, _, sts = chat(img, _context, None, params)
    print('<Assistant>:', _answer)
    return _answer


from flask import Flask, request, jsonify
from PIL import Image
app = Flask(__name__)


def dict_value(d, name, default=None, pick_first=False):
    if name in d:
        if pick_first:
            return d[name]#[0]
        else:
            return d[name]
    return default


@app.route('/chat', methods=["POST", "GET"])
def chat_r():
    file = request.files['file']
    data = request.form.get('data')
    if file and data:
        data = json.loads(data)
        image = Image.open(file)
        question = data['question']
        repetition_penalty = dict_value(data, 'repetition_penalty', 1.2)
        top_k = dict_value(data, 'top_k', 3)
        top_p = dict_value(data, 'top_p', 0.85)
        temperature = dict_value(data, 'temperature', 0.6)
        max_new_token = dict_value(data, 'max_new_token', 896)
        _answer = respond(question, image, repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p,
                temperature=temperature, max_new_token=max_new_token )
        return jsonify({'code': 0, 'data': _answer})
    else:
        return jsonify({'code': -1, 'data': 'param error'})


@app.route("/check", methods=["POST", "GET"])
def check():
    return jsonify({'code': 0})


def setup_as_service():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=35167)
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--lora', type=str, default='/lora')
    args = parser.parse_args()
    model_path = args.model_dir
    if not args.model_dir:
        if os.path.exists('/local_model_path.txt'):
            with open('/local_model_path.txt', 'r', encoding='utf8') as f:
                model_path = f.read()
    if model_path:
        # m_path = args.model_dir
        init_model(model_path, args.lora)

    app.run(host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    setup_as_service()


