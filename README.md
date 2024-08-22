# MarCom-Core
MarCom-Core is a repository containing core functions needed by the [MarCom](https://github.com/Hanyue-s-FYP) project

## Implementation & Features
- Implemented with LangChain, this core features 2 main features
    - Market Simulation with LLM backed agents to produce more understandable results, as LLMs are natural language oriented
    - Product Competitor Research, which transforms a product detail to a query for web search, performs the web search on DuckDuckGo, and passes to another LLM to generate research report based on the web search results
- LLMs are implemented with llama3.1

## Setup and running the project
> Ensure that you have python > 3.12 installed on your machine
### Install Ollama and pull llama3.1 model
- Install [Ollama](https://ollama.com/)
```sh
# Install llama3.1
ollama pull llama3.1
```

### Clone this repository to local
> No need to recurse submodules as a copy of the generated grpc files are attached in the repository and will be updated together with the repository
```sh
git clone https://github.com/Hanyue-s-FYP/MarCom-Core.git
```
### Create a virtual environment and activate it (encouraged)
```sh
py -m venv .venv
```
OR
```sh
python3 -m venv .venv
```
### Activate virtual environment
```sh
# MacOS/Linux
source .venv/bin/activate
# Windows
.venv/Scripts/activate
```
### Install dependencies
```sh
py -m pip install -r requirements.txt
```
OR
```sh
python3 -m pip install -r requirements.txt
```
### Setup environment
> Setup environment variables or add them in the .env file (reference .env.example, you can use the same value or define yours, the MODEL key does not do anything for now as only llama3.1 is supported)
### Run the main file
```sh
py main.py
```
OR
```sh
python3 main.py
```