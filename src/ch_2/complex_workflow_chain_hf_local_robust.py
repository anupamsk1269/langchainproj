# src/ch_2/complex_workflow_chain_hf_local_robust.py
import os
import sys
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables safely
load_dotenv(find_dotenv())

def validate_environment():
    """Validates that necessary tokens are present in the environment."""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("[WARNING] No Hugging Face token detected in environment variables. Model downloads may be throttled.")
    else:
        print("[INFO] Hugging Face authentication token successfully detected.")

def run_robust_hf_local_workflow():
    validate_environment()
    
    print("\n[INFO] Initializing local Hugging Face model pipeline (Qwen/Qwen2.5-1.5B-Instruct)...")
    try:
        # Initialize local transformers pipeline with robust configurations
        hf_pipeline = HuggingFacePipeline.from_model_id(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 256,
                "temperature": 0.2,
                "do_sample": True
            }
        )
    except Exception as e:
        print(f"[ERROR] Failed to load local model pipeline: {e}")
        sys.exit(1)

    try:
        # Wrap in ChatHuggingFace interface
        llm = ChatHuggingFace(llm=hf_pipeline)
        parser = StrOutputParser()

        # Step 1: Technical Root-Cause Prompt Template
        tech_prompt = PromptTemplate(
            input_variables=["domain", "issue"],
            template=(
                "You are a lead enterprise architect in {domain}. "
                "Analyze the following technical issue and provide a rigorous root-cause breakdown: {issue}."
            )
        )

        # Step 2: Executive Action Plan Prompt Template
        executive_prompt = PromptTemplate(
            input_variables=["domain", "issue", "technical_analysis"],
            template=(
                "Based on the domain '{domain}' and original issue '{issue}', review the root-cause analysis below:\n\n"
                "{technical_analysis}\n\n"
                "Now, write a concise 3-step executive remediation roadmap for senior leadership."
            )
        )

        # Build LCEL Sequential Chain
        assessment_chain = tech_prompt | llm | parser

        complete_workflow = (
            {"technical_analysis": assessment_chain, "domain": RunnablePassthrough(), "issue": RunnablePassthrough()}
            | executive_prompt
            | llm
            | parser
        )

        domain_input = "Oracle Cloud Enterprise Solutions"
        issue_input = "Metadata job failures causing duplicate records during data synchronization tracks"

        print(f"[INFO] Executing workflow for domain: '{domain_input}'...\n")
        
        result = complete_workflow.invoke({
            "domain": domain_input,
            "issue": issue_input
        })

        print("--- Final Executive Remediation Roadmap (Robust Local HF) ---")
        print(result)

    except Exception as e:
        print(f"[ERROR] An error occurred during chain execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_robust_hf_local_workflow()