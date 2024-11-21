#%%
import requests
import time
import logging
import re
from transformers import AutoTokenizer
#%%
API_TOKEN = 'YOUR_TOKEN_HERE'   
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
#%%
# pre-processing functions
def query(payload):
    "Query the Hugginface API"
    payload['inputs'].append({"role": "assistant", "content": "ANSWER: "})
    new_chat =  tokenizer.apply_chat_template(payload['inputs'], tokenize=False, continue_generation = True)
    payload['inputs'] = new_chat
    try:
        response = requests.post(API_URL, headers=headers, json=payload).json()
    except:
        return None
    return response

def get_system_prompt(text):
    system_message = {"role": "system", "content": text}
    return system_message

def get_user_prompt(text):
    user_message = {"role": "user", "content": text}
    return user_message

#%%

class ClaimCheck:
        def __init__(self):
            """
            Initializes the verifier. If a claims file is provided, it loads the contrarian claims from the file.
            Args:
                claims_file (str): Path to the CSV file with claim data.
            """

            self.contrarian_claims = {}
            self.params = {"do_sample": False,
                "max_new_tokens": 10,
                "return_full_text": False, 
                }

        
        # def extract_answer(self, response):
        #     # Extract the last line of the response
        #     generated_text = response[0]['generated_text']
        #     lines = generated_text.split('\n')
        #     last_line = lines[-1].strip()
        #     return last_line
        def get_text_from_query(self, payload):
            while True:
                new_payload = payload.copy()
                response = query(payload = new_payload)
                #print(response)
                # if response == None:
                #     continue
                # else:
                try:
                    return response[0]['generated_text']
                except:
                    continue


        def yes_or_no(self, prompt):
            while True:
                payload = {"inputs": prompt, "parameters": self.params, "options": {"use_cache": False}}
                answer = self.get_text_from_query(payload=payload)
                #print(answer)
                if 'yes' in answer.lower():
                    return 1
                elif 'no' in answer.lower():
                    return 0
                else:
                    continue

        def claim_check(self, text, question, topic=None):
            system_text = """ROLE: You are a concise assistant tasked with analyzing news articles. Your answer must only be "yes" or "no"."""
            if topic != None:
                system_text += f""" on the topic of {topic}"""
            system_prompt = get_system_prompt(system_text+ """. """)
            user_text = f"""TEXT: {text} \n\n TASK: Answer the following yes/no question. QUESTION: {question}."""
            prompt = [get_system_prompt(system_prompt), get_user_prompt(user_text)]
            response = self.yes_or_no(prompt)
            return response
            #print(response)
            #print(response)
            # if response == 'yes':
            #     return 1
            # return 0
        
