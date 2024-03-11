import os
from dotenv import load_dotenv
import json
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage,  SystemMessage
import openai

import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import notebook_login
from datasets import load_dataset
import subprocess


def read_file_generator(file_path):
    """
    Generator function to efficiently read a file line by line.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()

def split_data_into_chunks(data, chunk_size=2000, chunk_overlap=500):
    """
    Split the extracted data into fixed-size text chunks.

    Parameters:
        - data (list): List of dictionaries containing "page_content" and "metadata".
        - chunk_size (int): Size of each text chunk.
        - chunk_overlap (int): Overlap between adjacent chunks.

    Returns:
        - list: List of text chunks.
    """
    text_chunks = []
    
    for entry in data:
        content = entry.get("page_content", "")
        metadata = entry.get("metadata", {})
        
        for start in range(0, len(content), chunk_size - chunk_overlap):
            end = start + chunk_size
            text_chunk = content[start:end]
            
            text_chunks.append({
                "page_content": text_chunk,
                "metadata": metadata
            })

    return text_chunks

def save_to_jsonl(dataset_name, question_answer_list):
    """
    Save question-answer pairs with metadata to a JSONL file.

    Parameters:
        - dataset_name (str): The name of the dataset.
        - question_answer_list (list): List of dictionaries containing question, answer, and metadata.
    """
    file_name = f"/home/llmao/fastapi/DataGen/output/{dataset_name}.jsonl"

    with open(file_name, "a", encoding="utf-8") as file:
        for qa in question_answer_list:
            # Combine question, answer, and metadata into a dictionary
            entry = {
                "question": qa["question"],
                "answer": qa["answer"],
                "content": qa.get("content", ""),
                "text": f"[INST] <<SYS>> You are Shambu, a helpful, respectful, honest, and a Personal Healthcare assistant. Always answer as helpfully as possible, while being safe. If you don’t know the answer to a question, please don’t share false information. <</SYS>> {qa['question']} [/INST] {qa['answer']}"
            }
            # Write the dictionary to the JSONL file
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_text_chunk(text,dataset_name):
    json_response_format = [
        {
            "question": "In the context of ...",
            "answer": "..."
        },
        {
            "question": "In the context of ...",
            "answer": "..."
        },
        {
            "question": "In the context of ...",
            "answer": "..."
        }
    ]
    # Short Response    
    short_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at qurating/generating questions and answers from a given piece of text.
                            The questions and answers you generate are unique from one another and are not repeated.
                            You always respond in the following json format ```question_answer:{json_response_format}```"""
            },
            {
                "role": "user",
                # you should add the summary and modify the prompt to your liking
                "content": f"""given the context which is about *summary of the document* 
                \n
                {text['page_content']} 
                \n
                provide 5 important questions and answers pairs base on the text above , 
                The Question must begin with "In the context of ...\".The answer borrow, verbatim, from the text above. 
                In providing each question consider that the reader does not see or have access to any of the other questions from context. 
                Vary the style and format of questions. Let the answers be descriptive around 100 to 200 words
                """
                # Respond in only JSON following this format and nothing else {json_response_format}
            }
        ]
    )

    try:
        short_response_output = eval(short_response.choices[0].message.content)
        # Check if the format is correct and meets the criteria
        if (
            isinstance(short_response_output, dict)
            and "question_answer" in short_response_output
            and isinstance(short_response_output["question_answer"], list)
            and len(short_response_output["question_answer"]) >= 3
            and all(
                isinstance(qa, dict)
                and "question" in qa
                and "answer" in qa
                for qa in short_response_output["question_answer"]
            )
        ):
            # Add metadata to each question-answer pair
            for qa in short_response_output["question_answer"]:
                qa["content"] = f"{text['page_content']}"
                qa["metadata"] = text['metadata']

            # print("Short response format is correct.")
        else:
            print("Short response format is incorrect. Running the query again.")
            

    except Exception as e:
        print("Error in short_response_output", e)

    # Long Response
    long_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at qurating/generating questions and answers from a given piece of text.
                            The questions and answers you generate are unique from one another and are not repeated.
                            You always respond in the following json format ```question_answer:{json_response_format}```"""
            },
            {
                "role": "user",
                "content": f"""given the context which is about Tao Science which is written by "Rulin Xiu" and "Zhi Gang Sha"
                \n
                {text['page_content']} 
                \n
                provide 4 important questions and answers pairs base on the text above , 
                The Question must begin with "In the context of...\".The answer borrow, verbatim, from the text above. 
                In providing each question consider that the reader does not see or have access to any of the other questions from context. 
                Vary the style and format of questions. Let the answers be descriptive and lengthy.
                The answer should at least be 1000 words
                """
                # Respond in only JSON following this format and nothing else {json_response_format}
            }
        ]
    )

    try:
        long_response_output = eval(long_response.choices[0].message.content)

        # Check if the format is correct and meets the criteria
        if (
            isinstance(long_response_output, dict)
            and "question_answer" in long_response_output
            and isinstance(long_response_output["question_answer"], list)
            and len(long_response_output["question_answer"]) >= 2
            and all(
                isinstance(qa, dict)
                and "question" in qa
                and "answer" in qa
                for qa in long_response_output["question_answer"]
            )
        ):
            # Add metadata to each question-answer pair
            for qa in long_response_output["question_answer"]:
                qa["content"] = f"{text['page_content']}"
                qa["metadata"] = text['metadata']

            # print("Long response format is correct.")
        else:
            print("Long response format is incorrect. Running the query again.")
            

    except Exception as e:
        print("Error in long_response_output", e)
    save_to_jsonl(dataset_name, short_response_output["question_answer"])
    save_to_jsonl(dataset_name, long_response_output["question_answer"])


def huggingface_upload(dataset_name):
    hf_token = "YOUT_API_KEY_HERE"
    subprocess.run(["huggingface-cli", "login", "--token", hf_token])
    dataset = load_dataset('json', data_files=f"/home/llmao/fastapi/DataGen/output/{dataset_name}.jsonl")
    dataset
    dataset.push_to_hub(f"LLMao/{dataset_name}")
    print("successfully pushed to HF")



def process_and_store_data(dataset_name):
    # Set the OpenAI API key
    openai.api_key = "YOUR_API_KEY_HERE"

    """
    Read content from files in a folder, ignore empty documents, and generate data.
    """

    data = []
    for filename in os.listdir("/home/llmao/fastapi/DataGen/corpus"):
        file_path = os.path.join("/home/llmao/fastapi/DataGen/corpus", filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            # Ignore empty documents
            document = {
                "page_content": "", 
                "metadata": {"source": file_path, "page": 0},
            }

            # Read content from the file
            document["page_content"] = "\n".join(read_file_generator(file_path))

            data.append(document)

    # Save the resulting data to a file
    # output_file_path = f"/home/llmao/fastapi/DataGen/output/{dataset_name}.jsonl"
    # with open(output_file_path, "w", encoding="utf-8") as output_file:
    #     for entry in data:
    #         output_file.write(json.dumps(entry) + "\n")
    
    print(data[:1])

    if(data == ""):
        print("error parsing the files , or no files parsed . \n exiting function ")
        return 

    chunk_size = 2500
    chunk_overlap = 500
    text_chunks = split_data_into_chunks(data, chunk_size, chunk_overlap)
    print("Length of the whole documentation is:", len(text_chunks),end="\n")

    print(text_chunks[1]['page_content'],end="\n")

    # Set the number of parallel processes
    num_processes = 4  #optimal performance without crashing !!

    # # Use a ThreadPoolExecutor for parallel execution
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
    #     # Define the function to be executed in parallel
    #     future_to_chunk = {executor.submit(process_text_chunk, text_chunk, dataset_name): text_chunk for text_chunk in text_chunks}

    #     # Use tqdm to track progress
    #     for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(text_chunks), desc="Processing Text Chunks"):
    #         pass  # Processing happens in parallel, tqdm just tracks completion
    print("initializing the model to generate structured data")

    for text_chunk in text_chunks:
        process_text_chunk(text_chunk,dataset_name)
        
    print(f"structured_data generated , stored in {dataset_name} , uploading to huggingface")
    huggingface_upload(dataset_name)
