from tools.read_json import read_llm_prompt_json

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

class LLM:
    def __init__(self, model_name: str, provider: str, is_chat: bool, temperature: float = 0.2):
        self.is_chat = is_chat
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.prompt_system = read_llm_prompt_json(self.model_name)
        self.prompt_template = None
        self.openai_key = ""
        if self.provider == "OpenAI":
            try:
                load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)),".env"))
                self.openai_key = os.getenv("OPENAI_API_KEY")
            except:
                print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in the .env file.")
        self.model = None
        self.set_model()
        self.parser = StrOutputParser()

    def set_model(self):
        if self.provider == "OpenAI":
            if self.is_chat:
                from langchain_openai import ChatOpenAI
                self.model = ChatOpenAI(openai_api_key=self.openai_key, model_name=self.model_name, temperature=self.temperature)
                self.prompt_template = ChatPromptTemplate.from_messages([
                    ("system", self.prompt_system),
                    ("user", "{content}")
                ])
            else:
                from langchain_openai import OpenAI
                self.model = OpenAI(openai_api_key=self.openai_key, model_name=self.model_name, temperature=self.temperature)
                if not "{content}" in self.prompt_system:
                    self.prompt_system += "\n{content}"
                self.prompt_template = PromptTemplate.from_template(self.prompt_system)

        elif self.provider == "HuggingFace":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
            from langchain_huggingface import HuggingFacePipeline

            # check if NVIDIA GPU is available
            print(f"LLM {self.model_name} runs on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
            # Add HF_HUB_DISABLE_SYMLINKS_WARNING environment variable to avoid warning
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            # Set the seed for reproducibility
            torch.random.manual_seed(0)

            double_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=double_quant_config)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, do_sample=True, temperature=self.temperature, return_full_text=False)

            self.model = HuggingFacePipeline(pipeline=pipe)
            if not "{content}" in self.prompt_system:
                self.prompt_system += "\n{content}"
            self.prompt_template = PromptTemplate.from_template(self.prompt_system)
    
    def run(self, prompt: str):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain.invoke({"content": prompt})