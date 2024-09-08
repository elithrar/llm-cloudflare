## cloudflare-llm

> 🚧 Work in progress

A plugin for the [`llm`](https://llm.datasette.io/en/stable/) CLI that allows you to use the text generation models (LLMs) running on Cloudflare [Workers AI](https://developers.cloudflare.com/workers-ai/models/#text-generation).

## Usage

```sh
# Install the plugin
llm install llm-cloudflare

# Provide a valid Workers AI token
# Docs: https://developers.cloudflare.com/workers-ai/get-started/rest-api/#1-get-api-token-and-account-id
llm keys set cloudflare

# Set your Cloudflare account ID
# Docs: https://developers.cloudflare.com/workers-ai/get-started/rest-api/#1-get-api-token-and-account-id
export CLOUDFLARE_ACCOUNT_ID="33charlonghexstringhere"
```

The default model is currently the [Llama 3.1 8B instruct](https://developers.cloudflare.com/workers-ai/models/llama-3.1-8b-instruct) model

## Model library

TODO

## License

Copyright Cloudflare, Inc (2024). Apache-2.0 licensed. See the LICENSE file for details.
