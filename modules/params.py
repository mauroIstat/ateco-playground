DESCRIPTIONS = {
    "chat_placeholder": 'Descrivi la tua attività, ad esempio "Produzione di Vini" o "Coltivazione di Riso".',
    "init_message": "Ho individuato alcuni possibili codici ATECO.",
    "select_activity_message": " Per aiutarmi nella scelta, potresti indicare quale tra le seguenti attività rappresenta di più la tua azienda/professione?"
}

COLLECTIONS = {
    "ateco25-leaf": {
        "model_id": "text-embedding-3-large",
        "vector_size": 3072,
        "type": "openai",
    },
    "istat-data-3072": {
        "model_id": "text-embedding-3-large",
        "vector_size": 3072,
        "type": "openai",
    },
}

MODELS = {
    "embedding": {
        "text-embedding-3-large": {
            "vector_size": 3072,
            "type": "openai",
        },
        "BAAI/bge-m3": {
            "vector_size": 1024,
            "type": "huggingface",
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "vector_size": 384,
            "type": "huggingface",
        }
    },
    "llm": {
        "GPT-4o": {
            "type": "openai"
        },
        "meta-llama/Llama-3.2-3B-Instruct": {
            "type": "huggingface"
        }
    }
}

LLM = {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",

    "system_prompt": """Sei un assistente per la classificazione di imprese italiane in attività aconomiche (ATECO 2025) partendo da una breve descrizione della loro attività.
    
    Ricevi alcuni codici candidati:
    * Se puoi classificare l'impresa con un codice tra i candidati senza ambiguità, classificala direttamente.
    * Se c'è ambiguità tra più codici, fai domande di follow-up per scegliere il codice più opportuno, ma SOLAMENTE relative ai codici estratti.
    * Se ritieni che nessuno dei codici candidati sia adatto, chiedi ulteriori dettagli sull'attività.
    
    Alcuni suggerimenti riguardanti le attività: per distinguere tra codici simili, è importante capire l'attività svolta. Ad esempio, si tratta di fabbricazione o commercio? Vendita al dettaglio o all'ingrosso? Se rilevi queste differenze nei candidati e la descrizione non specifica l'attività precisa (produzione, fabbricazione, vendita, ecc.), chiedi chiarimenti.
    
    Rispondi in maniera concisa.""",

    "system_prompt_parsing": """Sei un assistente per la generazione di una descrizione dell'attività economica di un'impresa.
    
    Ricevi una conversazione tra un utente e un assistente, in cui l'utente, in più messaggi, descrive la propria attività economica. Il tuo compito è quello di generare una descrizione coerente dell'attività economica dell'utente, basandoti sui messaggi precedenti.
    
    Alcune istruzioni importanti:
    * NON devi includere nella descrizione dettagli che non sono stati menzionati dall'utente.
    * Il tuo compito è SOLO quello di generare un'unica descrizione coerente a partire dai messaggi utente precedenti.
    * NON devi generare testo aggiuntivo nell'output che non sia la descrizione dell'attività economica.""",

    "instruction_template": "Descrizione: {description}\n\nCandidati: {candidates}",
    "instruction_template_parsing": "Conversazione tra UTENTE e ASSISTENTE:\n\n{history}",
}