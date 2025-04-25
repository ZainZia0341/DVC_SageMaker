import os
import torch
from dotenv import load_dotenv
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# from groq import Groq

MODEL_ID = "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit"
CACHE_DIR = "./my_model_cache"

load_dotenv()  # take environment variables from .env.


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# if "GROQ_API_KEY" not in os.environ:
#     print("Please set the GROQ_API_KEY environment variable.")


if "GOOGLE_API_KEY" not in os.environ:
    print("Please set the GOOGLE_API_KEY environment variable.")

# if "LANGCHAIN_API_KEY" not in os.environ:
#     print("Please set the LANGCHAIN_API_KEY environment variable.")

def load_llm():
    """Load and return the LLM pipeline wrapped for LangChain."""
    if not torch.cuda.is_available():
         print("No GPU found. Please ensure a GPU is available or modify the code for CPU usage.")
    else:
        print("Loading 4-bit model (Qwen 7B) With GPU...")
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256
        )
        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
        print("LLM loaded successfully!")
        return llm

def load_embedding_model():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embedding_model



def load_groq_model():
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model= "meta-llama/llama-4-scout-17b-16e-instruct", # meta-llama/llama-4-scout-17b-16e-instruct, # qwen-2.5-Coder-32B, # qwen-2.5-32B, # "deepseek-r1-distill-qwen-32b", # "deepseek-r1-distill-llama-70b", # "llama-3.3-70b-versatile", # "llama-3.1-70b-versatile", # "llama-3.3-70b-specdec", # "llama3-8b-8192"
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    return llm

def load_google_gemini_model():
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-pro-exp-02-05", # "gemini-2.5-pro-preview-03-25", # "Gemini-2.0-Flash", # "gemini-1.5-pro", "gemini-2.0-pro-exp-02-05"
        temperature=0,
        max_tokens=None,
    )
    return llm


# def groq_client_llm():
#     completion = groq_client.chat.completions.create(
#             model="meta-llama/llama-4-scout-17b-16e-instruct",
#             messages=[message],
#             temperature=0.7,
#             top_p=1,
#             stream=False,
#             stop=None
#     )
#     return completion


from .sagemaker_llm import SageMakerLLM

def load_sagemaker_model():
    endpoint = os.getenv("SAGEMAKER_ENDPOINT_NAME")
    if not endpoint:
        raise ValueError("Set SAGEMAKER_ENDPOINT_NAME in your env")
    return SageMakerLLM(endpoint_name=endpoint)