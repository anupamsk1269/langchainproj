# src/ch_2/complex_workflow_chain_hf.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())

def run_hf_complex_workflow():
    # Initialize the free Hugging Face chat model via serverless inference
    hf_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        temperature=0.2,
        max_new_tokens=256,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    llm = ChatHuggingFace(llm=hf_endpoint)
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

    # Build an LCEL Sequential Chain
    assessment_chain = tech_prompt | llm | parser

    complete_workflow = (
        {"technical_analysis": assessment_chain, "domain": RunnablePassthrough(), "issue": RunnablePassthrough()}
        | executive_prompt
        | llm
        | parser
    )

    domain_input = "Oracle Cloud Enterprise Solutions"
    issue_input = "Metadata job failures causing duplicate records during data synchronization tracks"

    print(f"Executing Hugging Face complex workflow for domain: '{domain_input}'...\n")
    
    result = complete_workflow.invoke({
        "domain": domain_input,
        "issue": issue_input
    })

    print("--- Final Executive Remediation Roadmap (Hugging Face) ---")
    print(result)

if __name__ == "__main__":
    run_hf_complex_workflow()