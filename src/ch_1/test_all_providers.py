# src/ch_1/test_all_providers.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv(find_dotenv())

def test_providers():
    print("Testing OpenAI...")
    openai_llm = ChatOpenAI(model="gpt-4o-mini")
    print("OpenAI response:", openai_llm.invoke("Say 'OpenAI active'").content)

    print("Testing Anthropic...")
    anthropic_llm = ChatAnthropic(model="claude-sonnet-4-6")
    print("Anthropic response:", anthropic_llm.invoke("Say 'Anthropic active'").content)

    print("Testing Gemini...")
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    print("Gemini response:", gemini_llm.invoke("Say 'Gemini active'").content)

    print("Testing Hugging Face...")
    hf_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    hf_llm = ChatHuggingFace(llm=hf_endpoint)
    print("Hugging Face response:", hf_llm.invoke("Say 'Hugging Face active'").content)

if __name__ == "__main__":
    test_providers()
