python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements_nv.txt
python FullTextScreener.py 

huggingface-cli login

watch -n 1 nvidia-smi

install conda =>miniconda3 (with document) : https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


git clone the project

install neo4j(7474-7687)

wikiqa(dataset)

pip install -r Requirements/requirements_nv.txt

conda create --name env_name python=3.11
conda activate env_name

docker run --name neo4j-apoc     --publish=7474:7474 --publish=7687:7687     --env NEO4J_AUTH=neo4j/neo4j_rag_poc     -e NEO4J_apoc_import_file_enabled=true     -e NEO4J_apoc_import_file_use__neo4j__config=true     -e NEO4J_PLUGINS='["apoc"]'     -v ./neo4j_vol1/data:/data     -v ./neo4j_vol1/plugins:/plugins     neo4j:latest