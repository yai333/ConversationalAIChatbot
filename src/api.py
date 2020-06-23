import json
import logging
import sys
import boto3
import random
import os
sys.path.insert(1, '/mnt/libs/py-libs')
import torch
import torch.nn.functional as F
from simpletransformers.conv_ai.conv_ai_utils import get_dataset
from simpletransformers.conv_ai import ConvAIModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

dynamodb = boto3.client('dynamodb')
TABLE_NAME = os.environ["CHAT_HISTORY_TABLE"]
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
history = []
convAimodel = ConvAIModel("gpt", "/mnt/libs/convai-model", use_cuda=False)
character = [
    "i like computers .",
    "i like reading books .",
    "i like talking to chatbots .",
    "i love listening to classical music ."
]


def get_chat_histories(userid):
    response = dynamodb.get_item(TableName=TABLE_NAME, Key={
        'userid': {
            'S': userid
        }})

    if 'Item' in response:
        return json.loads(response["Item"]["history"]["S"])
    return {"history": []}


def save_chat_histories(userid, history):
    return dynamodb.put_item(TableName=TABLE_NAME, Item={'userid': {'S': userid}, 'history': {'S': history}})


def sample_sequence(aiCls, personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args["max_length"]):
        instance = aiCls.build_input_from_segments(
            personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(
            instance["input_ids"], device=aiCls.device).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["token_type_ids"], device=aiCls.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args["temperature"]
        logits = aiCls.top_filtering(
            logits, top_k=args["top_k"], top_p=args["top_p"])
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[
            1] if args["no_sample"] else torch.multinomial(probs, 1)
        if i < args["min_length"] and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    logging.warn(
                        "Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def interact(raw_text, model, personality, userid, history):
    args = model.args
    tokenizer = model.tokenizer
    process_count = model.args["process_count"]

    model._move_model_to_device()

    if not personality:
        dataset = get_dataset(
            tokenizer,
            None,
            args["cache_dir"],
            process_count=process_count,
            proxies=model.__dict__.get("proxies", None),
            interact=True,
        )
        personalities = [dialog["personality"]
                         for dataset in dataset.values() for dialog in dataset]
        personality = random.choice(personalities)
    else:
        personality = [tokenizer.encode(s.lower()) for s in personality]

    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(
            model, personality, history, tokenizer, model.model, args)
    history.append(out_ids)
    history = history[-(2 * args["max_history"] + 1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    save_chat_histories(userid, json.dumps({"history": history}))
    return out_text


def lambda_handler(event, context):
    try:
        userid = event['userid']
        message = event['message']
        history = get_chat_histories(userid)
        history = history["history"]
        response_msg = interact(message, convAimodel,
                                character, userid, history)

        return {
            'message': json.dumps(response_msg)
        }
    except Exception as ex:
        logging.exception(ex)
