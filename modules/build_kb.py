import pandas as pd
from semantic_search.data import build_corpus
from semantic_search.local import LocalKnowledgeBase

def build_kb(path: str, model_id: str) -> LocalKnowledgeBase:
    """
    Build a local knowledge base from a given path.

    Args:
        path (str): The path to the directory containing the data files.

    Returns:
        LocalKnowledgeBase: An instance of LocalKnowledgeBase containing the built corpus.
    """
    DESCRIPTOR: bool = """{title}"""

    ateco_df = pd.read_csv(path)

    descriptors = []
    for idx, row in ateco_df.iterrows():
        title = row["title"]
        description = row["description"]
        if pd.isna(description):
            description = ""
        if pd.isna(title):
            title = ""
        descriptors.append(DESCRIPTOR.format(title=title, description=description))
    
    corpus = build_corpus(
        texts=descriptors,
        ids=ateco_df.index,
        metadata=[{"code": c} for c in ateco_df["code"]],
    )

    return LocalKnowledgeBase(
        corpus,
        model_id=model_id,
        batch_size=64
    )

    