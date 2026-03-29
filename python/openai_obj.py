from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletion(BaseModel):
    id: str | None = None
    model: str
    created: int | None = None
    messages: list[Message]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None


class Completion(BaseModel):
    id: str | None = None
    model: str
    created: int | None = None
    prompt: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: dict


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: dict