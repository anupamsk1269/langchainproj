# src/ch_2/complex_workflow_chain_hf_local.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())

def run_hf_local_complex_workflow():
    print("Loading local Hugging Face model weights (this may take a moment on first run)...")
    
    # Initialize the model locally using a transformers pipeline wrapper
    # Using a compact open-weights model suited for local Apple Silicon execution
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
            "temperature": 0.2,
            "do_sample": True
        }
    )
    
    # Wrap in ChatHuggingFace to support conversational messaging interfaces
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

    # Build the same LCEL Sequential Chain workflow structure
    assessment_chain = tech_prompt | llm | parser

    complete_workflow = (
        {"technical_analysis": assessment_chain, "domain": RunnablePassthrough(), "issue": RunnablePassthrough()}
        | executive_prompt
        | llm
        | parser
    )

    domain_input = "Oracle Cloud Enterprise Solutions"
    issue_input = "Metadata job failures causing duplicate records during data synchronization tracks"

    print(f"Executing local Hugging Face complex workflow for domain: '{domain_input}'...\n")
    
    result = complete_workflow.invoke({
        "domain": domain_input,
        "issue": issue_input
    })

    print("--- Final Executive Remediation Roadmap (Local Hugging Face) ---")
    print(result)

if __name__ == "__main__":
    run_hf_local_complex_workflow()