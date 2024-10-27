# Project setup

Install dependencies
```bash
poetry install
```

### Rename "modify_me.env" to ".env"
1. Rename "rename_me.env" to ".env"
2. Insert the following API keys into .env

### OpenAI setup
1. Create an API key at https://platform.openai.com/account/api-keys
2. Save it as environment variable under `OPENAI_API_KEY`

### Pinecone setup
1. Create an API key at https://app.pinecone.io/
2. Save it as environment variable under `PINECONE_API_KEY`

### PromptLayer setup
1. Create an API key at https://promptlayer.com/home
2. Save it as environment variable under `PROMPTLAYER_API_KEY`

After setting any environment variables, you may need to reset you IDE to pick up the changes.
