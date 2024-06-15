import nest_asyncio
import os
nest_asyncio.apply()
from llama_parse import LlamaParse  # pip install llama-parse
from llama_index.core import SimpleDirectoryReader  # pip install llama-index

parser = LlamaParse(
    api_key="llx-GEfoCa8Avhd0PnN2qeFrtvFFTuToF4fAi44fr0aPTfXcFJlW",
    # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="text"  # "markdown" and "text" are available
)


current_path = os.getcwd()
print("Current working directory:", current_path)

file_extractor = {".pdf": parser}
reader = SimpleDirectoryReader(input_files=["META-1-1000.pdf"], file_extractor=file_extractor)

documents = reader.load_data()
print(documents)
