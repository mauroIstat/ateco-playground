import os
import pandas as pd
import torch
import string
from threading import Thread
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from semantic_search.data import build_corpus
from semantic_search.local import LocalKnowledgeBase

hf_token = os.getenv("HF_TOKEN")

def build_kb(path: str, model_id: str) -> LocalKnowledgeBase:
    DESCRIPTOR: bool = """{title}"""

    ateco_df = pd.read_csv(path)

    descriptors = []
    for idx, row in ateco_df.iterrows():
        title = row["title"]
        description = row["descriptor"]
        if pd.isna(description):
            description = ""
        if pd.isna(title):
            title = ""
        descriptors.append(DESCRIPTOR.format(title=title, description=description))
    
    corpus = build_corpus(
        texts=descriptors,
        ids=ateco_df.index,
        metadata=[{"code": c, "title": t} for c, t in zip(ateco_df["code"], ateco_df["title"])],
    )

    return LocalKnowledgeBase(
        corpus,
        model_id=model_id,
        batch_size=64
    )

def build_multivector_kb(path: str, model_id: str) -> LocalKnowledgeBase:
    df = pd.read_csv(path)
    codes = []
    texts = []
    titles = []
    parsed_descs = []

    for i, row in df.iterrows():
        descs = split_descriptor(row["descriptor"])

        for desc in descs:
            codes.append(row["code"])
            texts.append(desc)
            titles.append(row["title"])
            parsed_descs.append(parse_descriptor(row["descriptor"]))
    
    corpus = build_corpus(
        texts=texts,
        ids=list(range(len(texts))),
        metadata=[{"code": c, "title": t, "description": d} for c, t, d in zip(codes, titles, parsed_descs)],
    )

    return LocalKnowledgeBase(
        corpus,
        model_id=model_id,
        batch_size=64
    )

def split_descriptor(desc: str) -> List[str]:
    return desc.split("\n\n") if type(desc) == str else []

def parse_descriptor(desc: str) -> str:
    indices = list(string.ascii_lowercase)
    descs = split_descriptor(desc)

    parsed_desc = ""
    for i, desc in enumerate(descs):
        parsed_desc += f"{indices[i]}) {desc}\n\n"

    return parsed_desc.rstrip("\n")


class Llama:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    @staticmethod
    def parse_description(messages: List[str]) -> str:
        description = ""
        for m in messages:
            if m["role"] == "user":
                description += "[UTENTE]: " + f"{m['content']}\n"
            elif m["role"] == "assistant":
                description += "[ASSISTENTE]: " + f"{m['content']}\n"
        return description
    
    @staticmethod
    def parse_prompt(system: str, instruction: str) -> str:
        system_prompt = f"""<|start_header_id|>system<|end_header_id|>
        {system}
        <|eot_id|>""" if system else ""

        instruction_prompt = f"""<|start_header_id|>user<|end_header_id|>
        {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return system_prompt + instruction_prompt
    
    @staticmethod
    def parse_history(system: str, messages: List[str]) -> str:
        system_prompt = f"""<|start_header_id|>system<|end_header_id|>
        {system}
        <|eot_id|>""" if system else ""
        
        history = ""

        for m in messages:
            if m["role"] == "user":
                history += f"""<|start_header_id|>user<|end_header_id|>
                {m["content"]}<|eot_id|>"""

            elif m["role"] == "assistant":
                history += f"""<|start_header_id|>assistant<|end_header_id|>
                {m["content"]}<|eot_id|>"""

        if m["role"] == "user":
            history += "<|start_header_id|>assistant<|end_header_id|>"
        
        return system_prompt + history
    
    def generate_description(self, system: str, history: List[str], max_new_tokens: int = 100) -> str:
        conversation = self.parse_description(history)
        prompt = self.parse_prompt(system, conversation)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


    def stream_with_history(self, system: str, messages: List[str], max_new_tokens: int = 100):
        prompt = self.parse_history(system, messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        thread = Thread(target=self.model.generate, kwargs={
            "input_ids": inputs["input_ids"],
            "streamer": self.streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": 0.9,
        })
        thread.start()

        for chunk in self.streamer:
            yield chunk
        
        thread.join()


    def stream(self, system: str, instruction: str, max_new_tokens: int = 100):
        prompt = self.parse_prompt(system, instruction)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        thread = Thread(target=self.model.generate, kwargs={
            "input_ids": inputs["input_ids"],
            "streamer": self.streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": 0.9,
        })
        thread.start()

        for chunk in self.streamer:
            yield chunk
        
        thread.join()
