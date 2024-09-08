import llm
from llm import ModelError
import os
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from pydantic import Field, field_validator, model_validator
from typing import Optional, List

DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct"


@llm.hookimpl
def register_models(register):
    # Workers AI text generation models: https://developers.cloudflare.com/workers-ai/models/#text-generation
    register(WorkersAI("@cf/meta/llama-3.1-8b-instruct"))


class WorkersAIOptions(llm.Options):
    stream: Optional[bool] = Field(
        default=True, description="Stream responses from the Workers AI API."
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="TODO",
    )

    temperature: Optional[float] = Field(
        default=None,
        description="TODO",
    )

    top_p: Optional[float] = Field(
        default=None,
        description="TODO",
    )

    top_k: Optional[int] = Field(
        default=None,
        description="TODO",
    )
    # TODO: Define options for temperature, top_k and top_p


class WorkersAI(llm.Model):
    needs_key = "cloudflare"
    key_env_var = "CLOUDFLARE_API_TOKEN"
    model_id = "cloudflare"
    can_stream = True
    cloudflare_account_id = ""
    api_base_url = ""

    class Options(WorkersAIOptions): ...

    def __init__(self, model_id):
        self.model_id = model_id
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if account_id is None:
            raise ModelError(
                "You must set the CLOUDFLARE_ACCOUNT_ID environment variable with a valid Cloudflare account ID"
            )

        self.cloudflare_account_id = account_id
        self.api_base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/v1"

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        if conversation:
            for response in conversation.responses:
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": response.prompt.prompt,
                        },
                        {"role": "assistant", "content": response.text()},
                    ]
                )
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        client = OpenAI(api_key=self.get_key(), base_url=self.api_base_url)

        kwargs = {
            "model": self.model_id,
            "messages": self.build_messages(prompt, conversation),
            "stream": prompt.options.stream,
            "max_tokens": prompt.options.max_tokens or None,
        }

        try:
            if stream:
                with client.chat.completions.create(**kwargs) as stream:
                    for text in stream:
                        yield text.choices[0].delta.content
            else:
                completion = client.chat.completions.create(**kwargs)
                yield completion.choices[0].message.content
        # via https://github.com/openai/openai-python?tab=readme-ov-file#handling-errors
        except APIConnectionError as e:
            raise ModelError(f"Failed to connect to the server: {e.__cause__}")
        except RateLimitError as e:
            raise ModelError(
                f"API rate limit exceeded (status code: {e.status_code}): {e.__cause__}"
            )
        except APIStatusError as e:
            raise ModelError(f"API error (status code: {e.status_code}): {e.__cause__}")

    def __str__(self):
        return f"Cloudflare Workers AI: {self.model_id}"
