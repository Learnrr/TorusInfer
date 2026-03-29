
class RequestOutput:
    def __init__(
        self, 
        request_id: int, 
        sequence_id:int, 
        token_ids: list, 
        output_text: str, 
        generated_tokens: list
    ):
        self.request_id = request_id
        self.sequence_id = sequence_id
        self.token_ids = token_ids
        self.output_text = output_text
        self.generated_tokens = generated_tokens