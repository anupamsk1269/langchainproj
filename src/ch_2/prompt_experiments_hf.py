# src/ch_2/prompt_experiments_hf.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())

def run_hf_prompt_experiment():
    # Define a reusable prompt template
    template = (
        "You are an expert enterprise software consultant specializing in {domain}. "
        "Provide a concise, professional assessment for a project facing {challenge}. "
        "Tone should be {tone}. Keep it under {word_limit} words."
    )
    
    prompt = PromptTemplate(
        input_variables=["domain", "challenge", "tone", "word_limit"],
        template=template
    )

    # Format the prompt
    formatted_prompt = prompt.format(
        domain="Oracle Cloud ERP migrations",
        challenge="metadata job failures during data synchronization",
        tone="authoritative and pragmatic",
        word_limit=75
    )

    print("--- Formatted Prompt ---")
    print(formatted_prompt)
    print("\n-----------------------\n")

    # Initialize Hugging Face Endpoint with control parameters
    hf_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        temperature=0.2,
        max_new_tokens=150,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    
    # Wrap in LangChain Chat interface
    hf_llm = ChatHuggingFace(llm=hf_endpoint)

    print("Executing Hugging Face LLM call with customized parameters...")
    response = hf_llm.invoke(formatted_prompt)
    
    print("\n--- Model Response ---")
    print(response.content)

if __name__ == "__main__":
    run_hf_prompt_experiment()