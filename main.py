import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pinecone import Pinecone, PodSpec
from promptlayer import PromptLayer

promptlayer_client = PromptLayer(api_key=os.environ.get('PROMPTLAYER_API_KEY'))

openai = promptlayer_client.openai

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME = 'recipes'

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete the index if it exists
# if PINECONE_INDEX_NAME in pc.list_indexes().names():
#     pc.delete_index(PINECONE_INDEX_NAME)
#     print(f"Deleted existing index: {PINECONE_INDEX_NAME}")

# Create the index if it does not already exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric='euclidean',
        spec=PodSpec(environment="gcp-starter")
   )

index = pc.Index(PINECONE_INDEX_NAME)

recipe_filepaths = [DATA_DIR / file for file in os.listdir(DATA_DIR) if file.endswith('.txt')]

for filepath in recipe_filepaths:
    # Extract the filename without its extension from a given path
    base_name = os.path.basename(filepath)
    recipe_id, _ = os.path.splitext(base_name)

    # Read the content of a text file, convert it into a vector, insert it into the index
    with open(filepath, 'r') as file:
        recipe_content = file.read()
    
    recipe_embedding_response = openai.Embedding.create(
        input=recipe_content,
        engine='text-embedding-3-small',
    )
    vector = recipe_embedding_response['data'][0]['embedding']
    index.upsert(vectors=[(recipe_id, vector)])

food = 'chicken'

# Fetch IDs of recipes related to the given text using Pinecone
food_embedding_response = openai.Embedding.create(
    input=food,
    engine='text-embedding-3-small',
)
vector = food_embedding_response['data'][0]['embedding']

query_results = index.query(
    vector=vector,
    top_k=2,
    include_values=True
)

recipe_ids = {match['id'] for match in query_results['matches']}

# Retrieve recipe contents for a given list of recipe IDs
recipes_list = [open(DATA_DIR / f'{recipe_id}.txt', 'r').read() for recipe_id in recipe_ids]

"""
Given a prompt and a list of recipes, use GPT to suggest one of the recipes
"""

# Fetch our template from PromptLayer
recipe_template = promptlayer_client.prompts.get("recipe_template")
system_content = recipe_template['messages'][0]['prompt']['template']
user_content_template = recipe_template['messages'][1]['prompt']['template']

# Set our template variables
variables = {
    'food': food,
    'recipes_string': '\n\n'.join(recipes_list)
}

response, pl_request_id = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            'role': 'system',
            'content': system_content
        },
        {
            'role': 'user',
            'content': user_content_template.format(**variables)
        },
    ],
    temperature=0.5,
    max_tokens=1024,
    return_pl_id=True
)

print(response.choices[0].message.content)

# Associate request with a prompt template
promptlayer_client.track.prompt(
    request_id=pl_request_id,
    prompt_name='recipe_template',
    prompt_input_variables=variables
)

promptlayer_client.track.score(
    request_id=pl_request_id,
    score=(100 if food in response.choices[0].message.content else 0),
)

promptlayer_client.track.metadata(
    request_id=pl_request_id,
    metadata={
        "user_id": "abc123",
        "session_id": "xyz456",
        "is_vegetarian": "false",
    }
)