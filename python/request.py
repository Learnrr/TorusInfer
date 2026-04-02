
class Request:
    def __init__(self, sequence_id, prompt):
        self.sequence_id = sequence_id
        self.prompt = prompt
        self.length = 0
        self.token_ids = []

class APIRequest():
    def __init__(
        self, 
        prompt, 
        model, 
        temperature, 
        top_p, 
        top_k,
        max_tokens, 
        presence_penalty, 
        frequency_penalty, user
    ):
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.user = user