import promptflow
import streamlit as st

if __name__ == "__main__":
    pf = promptflow.PFClient()
    st.title("Product QA interface")
    query = st.text_input("Write your query here")
    print(query)
    if query:
        output = pf.flows.test(
            ".",
            inputs={
                "query": query,
            },
        )
        response = output['response']
        st.write(response)