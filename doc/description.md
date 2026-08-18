AI engineering Named entity recognition for document pipelines Assignment  
Due in 11d 
Build: A token-classification model that extracts persons, organizations, and locations, the core of any KYC or document-intelligence pipeline. 
Data: CoNLL-2003 (English) and MasakhaNER (Hugging Face; 10 African languages including Amharic). 
Core methods: BIO tagging, subword-to-word label alignment, entity-level F1 with seqeval. 
Why it's intermediate: Aligning labels across subword tokens trips up most first attempts, and you must evaluate at entity level rather than token accuracy, which flatters bad models. 
Due Thu 20 Aug, 09:00