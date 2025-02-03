import llm
from llm import ModelError
import os
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from pydantic import Field
from typing import Optional, List

DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct"


@llm.hookimpl
def register_models(register):
    # Workers AI text generation models: https://developers.cloudflare.com/workers-ai/models/#text-generation
    #
    # Generated via:
    # curl \
    # "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/models/search?per_page=1000" \
    # -H "Authorization: Bearer ${WORKERS_AI_TOKEN}" \
    # | jq --raw-output '.result[] | select (.task.name | contains("Text Generation")) | "register(WorkersAI(\"\(.name)\"))"'
    register(WorkersAI("@cf/qwen/qwen1.5-0.5b-chat"))
    register(WorkersAI("@cf/google/gemma-2b-it-lora"))
    register(WorkersAI("@hf/nexusflow/starling-lm-7b-beta"))
    register(WorkersAI("@cf/meta/llama-3-8b-instruct"))
    register(WorkersAI("@cf/meta/llama-3.2-3b-instruct"))
    register(WorkersAI("@hf/thebloke/llamaguard-7b-awq"))
    register(WorkersAI("@hf/thebloke/neural-chat-7b-v3-1-awq"))
    register(WorkersAI("@cf/meta/llama-2-7b-chat-fp16"))
    register(WorkersAI("@cf/mistral/mistral-7b-instruct-v0.1"))
    register(WorkersAI("@cf/mistral/mistral-7b-instruct-v0.2-lora"))
    register(WorkersAI("@cf/tinyllama/tinyllama-1.1b-chat-v1.0"))
    register(WorkersAI("@hf/mistral/mistral-7b-instruct-v0.2"))
    register(WorkersAI("@cf/fblgit/una-cybertron-7b-v2-bf16"))
    register(WorkersAI("@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"))
    register(WorkersAI("@cf/thebloke/discolm-german-7b-v1-awq"))
    register(WorkersAI("@cf/meta/llama-2-7b-chat-int8"))
    register(WorkersAI("@cf/meta/llama-3.1-8b-instruct-fp8"))
    register(WorkersAI("@hf/thebloke/mistral-7b-instruct-v0.1-awq"))
    register(WorkersAI("@cf/qwen/qwen1.5-7b-chat-awq"))
    register(WorkersAI("@cf/meta/llama-3.2-1b-instruct"))
    register(WorkersAI("@hf/thebloke/llama-2-13b-chat-awq"))
    register(WorkersAI("@hf/thebloke/deepseek-coder-6.7b-base-awq"))
    register(WorkersAI("@cf/meta-llama/llama-2-7b-chat-hf-lora"))
    register(WorkersAI("@cf/meta/llama-3.3-70b-instruct-fp8-fast"))
    register(WorkersAI("@hf/thebloke/openhermes-2.5-mistral-7b-awq"))
    register(WorkersAI("@hf/thebloke/deepseek-coder-6.7b-instruct-awq"))
    register(WorkersAI("@cf/deepseek-ai/deepseek-math-7b-instruct"))
    register(WorkersAI("@cf/tiiuae/falcon-7b-instruct"))
    register(WorkersAI("@hf/nousresearch/hermes-2-pro-mistral-7b"))
    register(WorkersAI("@cf/meta/llama-3.1-8b-instruct"))
    register(WorkersAI("@cf/meta/llama-3.1-8b-instruct-awq"))
    register(WorkersAI("@hf/thebloke/zephyr-7b-beta-awq"))
    register(WorkersAI("@cf/google/gemma-7b-it-lora"))
    register(WorkersAI("@cf/qwen/qwen1.5-1.8b-chat"))
    register(WorkersAI("@cf/meta/llama-3-8b-instruct-awq"))
    register(WorkersAI("@cf/meta/llama-3.2-11b-vision-instruct"))
    register(WorkersAI("@cf/defog/sqlcoder-7b-2"))
    register(WorkersAI("@cf/microsoft/phi-2"))
    register(WorkersAI("@hf/meta-llama/meta-llama-3-8b-instruct"))
    register(WorkersAI("@hf/google/gemma-7b-it"))
    register(WorkersAI("@cf/qwen/qwen1.5-14b-chat-awq"))
    register(WorkersAI("@cf/openchat/openchat-3.5-0106"))

class WorkersAIOptions(llm.Options):
    stream: Optional[bool] = Field(
        default=True, description="Stream responses from the Workers AI API."
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to return in a response.",
    )

    temperature: Optional[float] = Field(
        default=None,
        description="'temperature' refers to a hyperparameter that controls the randomness and creativity of generated text. Higher temperatures produce more unpredictable and less coherent output, and lower temperatures produce more precise and factual but also more predictable and less imaginative text.",
    )

    top_p: Optional[float] = Field(
        default=None,
        description="'top_p' is a parameter that determines the probability threshold for generating a word or token, dictating how eagerly the model selects the next word in a sequence based on its predicted probability of being the next word. Higher values give more creative freedom and lower values lead to more coherent but less diverse outputs.",
    )

    top_k: Optional[int] = Field(
        default=None,
        description="'top-k' is a technique that restricts the model from generating a response by considering only the top-k most likely next tokens from the vocabulary, rather than considering all possible tokens. A higher value is less restrictive.",
    )


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
