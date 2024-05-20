1. Create environment variable using: python3 -m venv env
2. Install all dependencies: pip install -r requirements.txt
3. From terminal run: streamlit run app.py
4. Enter your query on UI and hit enter.
5. Just in case, open ai api key reaches rate limit error, replace with new one in inputs inside file flow.dag.yaml
6. The RAG triad results are stored in results folder. Take a look for your reference.
7. Docker image can be created and uploaded on repository using promptfow commands provided on the follwing link: https://microsoft.github.io/promptflow/how-to-guides/deploy-a-flow/deploy-using-docker.html