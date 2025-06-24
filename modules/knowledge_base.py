import pandas as pd
import numpy as np
from typing import List
from semantic_search.data import build_corpus
from semantic_search.local import LocalKnowledgeBase

def get_base(path: str, model_id: str) -> LocalKnowledgeBase:
    df = pd.read_csv(path)

    codes, texts, titles, descs, activities = [], [], [], [], []
    for idx, row in df.iterrows():
        desc = row["descriptor"]
        desc_list = split_descriptor(desc) if type(desc) == str else []

        descriptor_template = "#{title}\n{content}.\n\nPercorso: {hierarchy}"

        descriptors = [descriptor_template.format(title=row["title"], content=d, hierarchy=row["hierarchy"]) for d in desc_list]
        texts.extend(descriptors)

        for d in desc_list:
            codes.append(row["code"])
            titles.append(row["title"])
            descs.append(row["descriptor"])
            activities.append(row["activity"])
    
    corpus = build_corpus(
        texts=texts,
        ids=list(range(len(texts))),
        metadata=[{"code": c, "title": t, "description": d, "activity": a} for c, t, d, a in zip(codes, titles, descs, activities)]
    )

    return LocalKnowledgeBase(
        corpus=corpus,
        model_id=model_id,
        batch_size=64
    )


def split_descriptor(text: str) -> List[str]:
    elements = text.split("\n\n") if type(text) == str else []
    
    items = []
    for el in elements:
        if ":\n*" not in el:
            items.append(el.rstrip("\n"))

        else:
            lines = el.strip().splitlines()
            header = ""
            _items = []
            for line in lines:

                if line.startswith("*"):
                    item = line[1:].strip()

                    if header:
                        _items.append(f"{header} {item.lower()}")

                elif line:
                    header = line.rstrip(":")
                
            _items = [i.rstrip("\n") for i in _items]
            items.extend(_items)
    
    return items

def parse_retrieved(results):
    codes = [r.metadata["code"] for r in results[0]]
    titles = [r.metadata["title"] for r in results[0]]
    descriptions = [r.metadata["description"] for r in results[0]]
    activities = [r.metadata["activity"] for r in results[0]]
    texts = [r.text for r in results[0]]
    scores = [r.score for r in results[0]]

    df = pd.DataFrame({
        "code": codes,
        "title": titles,
        "matched_text": texts,
        "description": descriptions,
        "activity": activities,
        "score": scores
    })

    grouped_df = df.groupby("code").aggregate({
        "title": lambda x: np.unique(x)[0],
        "description": lambda x: np.unique(x)[0],
        "activity": lambda x: np.unique(x)[0],
        "score": "max",
    }).reset_index()

    return grouped_df.sort_values("score", ascending=False)

def parse_description(text: str) -> str:
    texts = text.split("\n\n")
    texts = ["\t* " + t for t in texts]
    clean_texts = []
    for t in texts:
        t = t.split("\n")
        clean_texts.append("\n\t\t".join(t))
    return "\n".join(clean_texts)