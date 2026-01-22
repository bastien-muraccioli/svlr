from tools.read_json import read_llm_prompt_json
from src.vlm import VLM
import time
import requests
import json

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


class LLM:
    def __init__(
        self, model_name: str, provider: str, is_chat: bool, temperature: float = 0.2, llm_is_vlm: bool = False
    ):
        self.llm_is_vlm = llm_is_vlm
        self.vlm = None
        if self.llm_is_vlm:
            self.vlm = VLM(vlm_name=model_name)
        self.is_chat = is_chat
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.prompt_system = read_llm_prompt_json(self.model_name)
        self.prompt_template = None
        self.openai_key = ""
        if self.provider == "OpenAI":
            try:
                load_dotenv(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
                )
                self.openai_key = os.getenv("OPENAI_API_KEY")
            except:
                print(
                    "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in the .env file."
                )
        self.model = None
        self.set_model()
        self.parser = StrOutputParser()

    def set_model(self):
        if self.provider == "OpenAI":
            if self.is_chat:
                from langchain_openai import ChatOpenAI

                self.model = ChatOpenAI(
                    openai_api_key=self.openai_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                )
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", self.prompt_system), ("user", "{content}")]
                )
            else:
                from langchain_openai import OpenAI

                self.model = OpenAI(
                    openai_api_key=self.openai_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                )
                if not "{content}" in self.prompt_system:
                    self.prompt_system += "\n{content}"
                self.prompt_template = PromptTemplate.from_template(self.prompt_system)

        elif self.provider == "Ollama":
            # from langchain_ollama.llms import OllamaLLM
            # if not self.llm_is_vlm:
            #     self.model = OllamaLLM(
            #         model=self.model_name,
            #         temperature=self.temperature,
            #     )
            if self.is_chat:
                self.prompt_template = ChatPromptTemplate.from_messages(
                    [("system", self.prompt_system), ("user", "{content}")]
                )
            else:
                if not "{content}" in self.prompt_system:
                    self.prompt_system += "\n{content}"
                self.prompt_template = PromptTemplate.from_template(self.prompt_system)

        elif self.provider == "HuggingFace":
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
                BitsAndBytesConfig,
            )
            from langchain_huggingface import HuggingFacePipeline

            # check if NVIDIA GPU is available
            # print(
            #     f"LLM {self.model_name} runs on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
            # )
            # # Add HF_HUB_DISABLE_SYMLINKS_WARNING environment variable to avoid warning
            # os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            # Set the seed for reproducibility
            torch.random.manual_seed(0)

            double_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=double_quant_config,
                # device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=100,
                do_sample=True,
                temperature=self.temperature,
                return_full_text=False,
            )
            self.model = HuggingFacePipeline(pipeline=pipe)

            if not "{content}" in self.prompt_system:
                self.prompt_system += "\n{content}"
            self.prompt_template = PromptTemplate.from_template(self.prompt_system)

    def run(self, prompt: str):
        if self.llm_is_vlm or self.provider == "Ollama":
            full_prompt = self.prompt_system.replace("{content}", prompt)
            start = time.time()
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": self.model_name,
                "prompt": full_prompt,
                "keep_alive": 0,
            }, stream=True)
            output = ""
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            output += data.get("response", "")
                        except json.JSONDecodeError:
                            print("Could not decode:", line)
                # print("Answer:", output.strip())
            else:
                print("Request failed with status", response.status_code)
                print(response.text)
            end = time.time()
            print(f"LLM: {self.model_name} Inference time = {end - start}s")
            # print(f"LLM Prompt = {full_prompt}")
            # print(f"LLM Response = {output.strip()}")
            return output.strip()
        else:
            chain = self.prompt_template | self.model | StrOutputParser()
            return chain.invoke({"content": prompt})