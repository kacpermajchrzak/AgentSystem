import torch
from transformers import pipeline

class LLM:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        self.model = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def check_if_news_is_fake(self, message, fact=None):
        if fact:
            message = f"{message} The fact is: {fact}."
    
        system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an assistant that receives short news statements. Your task is to determine whether the statement is factually correct or not. Respond with exactly one word: - Use "TRUE" if the statement is accurate. - Use "FAKE" if the statement is not accurate. Do not explain or justify the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>{message.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
        result = self.model(
            system_prompt,
            max_new_tokens=64,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=128001
        )
    
        response = result[0]["generated_text"]
        response = response.split("<|eot_id|>")[-1].strip()
    
        if "TRUE" in response:
            return False
        elif "FAKE" in response:
            return True
        
    def spread_the_news(self, message):
        system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an assistant that receives short news statements from the user. Your task is to rewrite the statement in your own words while preserving its original meaning. Do not evaluate or judge whether the news is  true or false — just rephrase it. Be concise, clear and answer only with rephrased news.<|eot_id|><|start_header_id|>user<|end_header_id|>{message.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
        result = self.model(
            system_prompt,
            max_new_tokens=64,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=128001
        )
    
        response = result[0]["generated_text"]
        response = response.split("<|end_header_id|>")[-1].strip()
        print(response)
        return response
