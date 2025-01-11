import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Carga las variables de entorno desde el archivo.env
load_dotenv()

# Obtiene las variables de entorno
HF_TOKEN = os.getenv('HF_TOKEN')
MODEL_NAME = os.getenv('MODEL_NAME')
TEMPERATURE = float(os.getenv('TEMPERATURE'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS'))
TOP_P = float(os.getenv('TOP_P'))

# Crea un cliente de inferencia con el token de Hugging Face
client = InferenceClient(api_key=HF_TOKEN)

# Define los mensajes de entrada
messages = [
    { "role": "user", "content": "Tell me a story" }
]

# Crea una solicitud de completado de texto
stream = client.chat.completions.create(
    model=MODEL_NAME, 
    messages=messages, 
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    stream=True
)

# Imprime los resultados
for chunk in stream:
    print(chunk.choices[0].delta.content)