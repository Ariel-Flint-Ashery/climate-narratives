#%%
import requests
import time
import logging
import re
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
from transformers import AutoTokenizer
#%%
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
    #print(new_chat)
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

class FrameExtract:
    def __init__(self, claims_file=None, log_name='newlog'):
        """
        Initializes the verifier. If a claims file is provided, it loads the contrarian claims from the file.
        Args:
            claims_file (str): Path to the CSV file with claim data.
        """
        fh = logging.FileHandler(f'{log_name}_FrameExtraction.log')
        fh.setLevel(logging.DEBUG) # or any level you want
        logger.addHandler(fh)
        self.contrarian_claims = {}
        if claims_file:
            self.contrarian_claims = claims_file
        self.short_params = {"do_sample": False,
            "max_new_tokens": 2,
            "return_full_text": False, 
            }
        self.long_params = {"do_sample": True,
                            "temperature": 0.2,
                "max_new_tokens": 300,
                "return_full_text": False, 
                }
        self.medium_params = {"do_sample": True,
                            "temperature": 0.2,
            "max_new_tokens": 100,
            "return_full_text": False, 
            }
        
    def get_text_from_query(self, payload):       
        while True:
            new_payload = payload.copy()
            response = query(payload = new_payload)
            try:
                return response[0]['generated_text']
            except:
                continue

    # def extract_integer(response):
    # """
    # Extracts the integer value of YOUR_ANSWER from the response.
    # The expected format is {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}.
    # """
    # # Use a regular expression to find the pattern {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}
    # match = re.search(r"{'ANSWER': (\d+); 'REASON': .+}", response)

    def get_thoughts(self, prompt, params):
        payload = {"inputs": prompt, "parameters": params, "options": {"use_cache": False}}
        response = self.get_text_from_query(payload)
        return response

    def extract_solution_id(self, text):
        match = re.search(r'The best solution is (\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def generate_single_vote(self, prompt, params):
        payload = {"inputs": prompt, "parameters": params, "options": {"use_cache": False}}
        solution_id = None
        while True:
            response = self.get_text_from_query(payload)
            solution_id = self.extract_solution_id(f"{response}")
            if solution_id is None:
                continue
            elif solution_id < 4:
                return solution_id

    def extract_answer(response):
        match = re.search(r'The final answer is (yes|no)', response, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return None

    def generate_yn_thought(self, prompt, params):
        payload = {"inputs": prompt, "parameters": params, "options": {"use_cache": False}}
        while True:
            response = self.get_text_from_query(payload)
            print(f"{response}")
            print(len(f"{response}"))
            #answer = self.extract_answer(f"{response}")
            match = re.search(r'The final answer is (yes|no)', response, re.IGNORECASE)
            if match:
                if "yes" in match.group(1).lower() or "no" in match.group(1).lower():
                    return response
            else:
                continue
    
    def extract_yes_no_answer(self, response):
        #answer = self.extract_answer(f"{response}")
        answer = re.search(r'The final answer is (yes|no)', response, re.IGNORECASE).group(1).lower()
        if "yes" in answer:
            return 1
        elif "no" in answer: 
            return 0

    def CoT_detect_framing(self, article, claim):

        question = f"""Does the TEXT apply a framing effect regarding the following claim: {claim}?"""
        system_text = """KNOWLEDGE: According to Robert Entmen's definition of framing, "to frame is to select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described".
        INSTRUCTION: Assess if specific language choices, focus points, or implied assumptions in the following TEXT promote a distinct perspective or argument, indicative of a frame.""" #+ """ Your answer must be in the following format {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}."""
        #TASK: Determine whether the following TEXT from a news article on climate change exhibits a framing effect.
        #   """
        
        user_text = f"""TEXT: \n\n {article} \n\n Answer the following yes/no question by reasoning step-by-step. QUESTION: {question}"""+ """ Conclude in the last line "The final answer is {s}", where s is "yes" or "no"."""
        prompt = [get_system_prompt(system_text), get_user_prompt(user_text)]
        thoughts = self.generate_yn_thought(prompt, params=self.long_params)
        
        return thoughts
    
    def hivemind_detection(self, article, claim, thought_list):
        question = f"""Does the TEXT apply a framing effect regarding the following claim: {claim}?"""
        all_agents = []
        for i, thought in enumerate(thought_list):
            agent_thought = f""" Agent {i+1}: {thought}"""
            all_agents.append(agent_thought)

        all_agents_text = """ \n""".join(all_agents)

        system_text = """KNOWLEDGE: According to Robert Entmen's definition of framing, "to frame is to select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described".
        INSTRUCTION: Assess if specific language choices, focus points, or implied assumptions in the following TEXT promote a distinct perspective or argument, indicative of a frame."""# +""" Your answer must be in the following format {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}."""
        text =  f"""\n\n TEXT: \n\n {article} \n\n"""
        agents_text = f""" These are the responses of other agents: {all_agents_text}. """
        user_text = text + agents_text + f""" Using the solutions from other agents as additional information answer the following yes/no question by reasoning step-by-step. QUESTION: {question}""" + """ Conclude in the last line "The final answer is {s}", where s is "yes" or "no"."""
        prompt = [get_system_prompt(system_text), get_user_prompt(user_text)]
        response = self.generate_yn_thought(prompt, params=self.long_params)
        final_response = self.extract_yes_no_answer(response)

        return final_response



    def CoT_explain_framing(self, article, claim):

        question = f"""How does the TEXT apply a framing effect regarding the following claim: {claim}?"""
        system_text = """KNOWLEDGE: According to Robert Entmen's definition of framing, "to frame is to select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described".
        INSTRUCTION: Assess how specific language choices, focus points, or implied assumptions in the following TEXT promote a distinct perspective or argument, indicative of a frame."""# + """ Your answer must be in the following format {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}."""
        #TASK: Determine whether the following TEXT from a news article on climate change exhibits a framing effect.
        #   """
        
        user_text = f"""TEXT: \n\n {article} \n\n Answer the following question by reasoning step-by-step. QUESTION: {question}."""
        prompt = [get_system_prompt(system_text), get_user_prompt(user_text)]
        thoughts = self.get_thoughts(prompt, params=self.long_params)
        
        return thoughts
        
    def update_explanation(self, article, claim, thought_list):

        question = f"""How does the TEXT apply a framing effect regarding the following claim: {claim}?"""
        all_agents = []
        for i, thought in enumerate(thought_list):
            agent_thought = f""" Agent {i+1}: {thought}"""
            all_agents.append(agent_thought)

        all_agents_text = """ \n""".join(all_agents)


        system_text = """KNOWLEDGE: According to Robert Entmen's definition of framing, "to frame is to select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described".
        INSTRUCTION: Assess if specific language choices, focus points, or implied assumptions in the following TEXT promote a distinct perspective or argument, indicative of a frame."""# + """ Your answer must be in the following format {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}."""
        text =  f"""\n\n TEXT: \n\n {article} \n\n"""
        agents_prompt = f""" These are the responses of other agents: {all_agents_text}. """
        user_text = text + agents_prompt + f""" Using the INSTRUCTION and the solutions from other agents as additional information give an updated answer to the following question. QUESTION: {question}"""
        prompt = [get_system_prompt(system_text), get_user_prompt(user_text)]
        thoughts = self.get_thoughts(prompt, params=self.long_params)
        
        return thoughts
        
    def hivemind_choose_best_explanation(self, article, claim, thought_list):

        all_agents = []
        for i, thought in enumerate(thought_list):
            agent_thought = f""" Solution {i+1}: {thought}"""
            all_agents.append(agent_thought)

        all_agents_text = """ \n""".join(all_agents)

        system_text = f"""KNOWLEDGE: According to Robert Entmen's definition of framing, "to frame is to select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described".
        INSTRUCTION: Assess how specific language choices, focus points, or implied assumptions in the following TEXT promote a distinct perspective or argument, indicative of a frame regarding the claim {claim}."""# + """ Your answer must be in the following format {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}."""
        text =  f"""\n\n TEXT: \n\n {article} \n\n"""
        agents_prompt = f""" These are possible SOLUTIONS: {all_agents_text}. """
        vote_prompt = """Given the possible SOLUTIONS for the INSTRUCTION, decide which solution is most promising. Analyze each solution in detail, then conclude in the last line "The best solution is {s}", where s the integer id of the solution."""
        user_text = text + agents_prompt + vote_prompt 
        prompt = [get_system_prompt(system_text), get_user_prompt(user_text)]
        response = self.generate_single_vote(prompt, params=self.long_params)

        return response


    def generate_frame_summary(self, article, claim, explanation):
        question = f" What specific frame is used in the TEXT to shape the interpretation of the following claim: {claim}?"
        system_text = f"""KNOWLEDGE: According to Robert Entmen's definition of framing, "to frame is to select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described"."""
        #+ """ INSTRUCTION: Your answer must be in the following format {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}."""#INSTRUCTION: Assess the specific language choices, focus points, or implied assumptions in the following TEXT promote a distinct perspective or argument, indicative of a frame regarding the claim {claim}"""
        text = f"TEXT: \n\n {article}"
        explanation_prompt = f"""This is an an EXPLANATION of how the TEXT applies a framing effect to the claim {claim}: {explanation}"""
        user_prompt = f""" \n\n Using the EXPLANATION, answer the following question using 3 to 10 words. QUESTION: {question}"""
        user_text = text + explanation_prompt + user_prompt
        prompt = [get_system_prompt(system_text), get_user_prompt(user_text)]
        thoughts = self.get_thoughts(prompt, params=self.medium_params)
        return thoughts
    
    def hivemind_choose_best_frame(self, article, claim, thought_list):

        all_agents = []
        for i, thought in enumerate(thought_list):
            agent_thought = f""" Solution {i+1}: {thought}"""
            all_agents.append(agent_thought)

        all_agents_text = """ \n""".join(all_agents)

        system_text = f"""KNOWLEDGE: According to Robert Entmen's definition of framing, "to frame is to select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described"."""
        #+ """ Your answer must be in the following format {'ANSWER': YOUR_ANSWER; 'REASON': YOUR_REASON}."""
        question = f"""QUESTION: What specific frame is used in the TEXT to shape the interpretation of the following claim: {claim}?"""
        text =  f"""\n\n TEXT: \n\n {article} \n\n"""
        agents_prompt = f""" These are possible SOLUTIONS: {all_agents_text}. """
        vote_prompt = f"""TASK: Given the possible SOLUTIONS, decide which solution is the most promising answer to the {question} """ + """Analyze each solution in detail, then conclude in the last line "The best solution is {s}", where s the integer id of the solution."""
        user_text = text + agents_prompt + vote_prompt 
        prompt = [get_system_prompt(system_text), get_user_prompt(user_text)]
        response = self.generate_single_vote(prompt, params=self.long_params)

        return response
        