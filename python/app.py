from flask import Flask, request, jsonify
import os
import torch
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"*": {"origins": "http://localhost:3000"}})

@app.route('/answer', methods=['POST'])
def get_answer():
    # Get the question from the request
    question = request.json['question']
    print("the question is ")
    print(question)
    MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://patelniraj313:JZW57Er7zqdKKCBR@cluster0.agb4hnr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    # initialize MongoDB python client
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

    DB_NAME = "langchain_db"
    COLLECTION_NAME = "test"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "index_name"

    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

    # The model that you want to train from the Hugging Face hub
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    # The instruction dataset to use
    # dataset_name = "mlabonne/guanaco-llama2-1k"
    dataset_name = "/content/drive/MyDrive/output.parquet"

    # Fine-tuned model name
    new_model = "llama-2-7b-miniguanaco"

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = "./results"

    # Number of training epochs
    num_train_epochs = 1

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training
    per_device_train_batch_size = 4

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 4

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 0

    # Log every X updates steps
    logging_steps = 25

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    # Change the default value of temperature using GenerationConfig class
    generation_config = GenerationConfig(
        temperature = 0.7,
    )

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        # generation_config=generation_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # insert the documents in MongoDB Atlas with their embedding
    vector_search = MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    def query_data(question):
        docs = vector_search.similarity_search(question, K=10)
        return docs

    # data from retriver
    def formatting_data(docs):
        formatted_data = [{"text": doc.page_content} for doc in docs]
        return formatted_data

    from flashrank import Ranker, RerankRequest

    # load the reranking model
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")

    def reranking(question, formatted_data):
        rerankrequest = RerankRequest(query=question, passages=formatted_data)
        results = ranker.rerank(rerankrequest)
        return results

    import collections
    text_list = collections.OrderedDict()

    def formatted_answer(results):
        answer = [item["text"] for item in results]
        for i in range(len(answer)):
            answer[i] = answer[i].replace("\n", " ")
        return answer[0]

    # question = "What is the responsibilty of democratic government?"

    docs = query_data(question)
    formatted_data = formatting_data(docs)
    reranked_results = reranking(question, formatted_data)
    question_content = formatted_answer(reranked_results)

    question_content

    def extract_answer(text):
        start_index = text.find("[/INST]") + len("[/INST]")
        # Extract the text from the start index to the end
        answer = text[start_index:]
        # Print the extracted text
        return answer

    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)

    prompt=f"""
    Using the provided information, answer the question in a comprehensive and informative way according to the given question content in 250 words.

    Question: {question}

    Question content: {question_content}

    """
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    answer = extract_answer(result[0]['generated_text'])

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)