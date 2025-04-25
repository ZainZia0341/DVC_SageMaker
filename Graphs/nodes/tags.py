from ..Graph_State import State
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..LLMs.llm import load_sagemaker_model

llm = load_sagemaker_model()


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

def tags_generation_node(state: State):
    # We'll assume the first message is user input
    question = state["messages"][-1].content
    print("------------------ tags generation node ---------------------")
    print(state)
    print(question)

    # Prompt: instruct the LLM to output exactly a JSON array with 5-10 lower-case tags
    prompt = PromptTemplate(
        template="""
            You are an assistant that must return exactly a JSON array of 5 to 10 lower-case tags 
            with no extra text, no triple backticks, and no keys. 
            For example:
            User Question: what are my top skills in my resume.
            AI Response:
            ["resume","skills","job","career","experience"]

            Do not include any additional explanations or formatting.

            The user question is: {question}

            Return only the JSON array. Nothing else.
        """,
        input_variables=["question"]
    )

    chain = prompt | llm | StrOutputParser()
    response_text = chain.invoke({"question": question})
    print("tags_generation_node => raw response:", response_text)

    # Parse the JSON to get the list of tags
    tags_list = []
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, list) and all(isinstance(t, str) for t in parsed):
            # Convert each tag to lower case to be safe
            tags_list = [t.lower() for t in parsed]
        else:
            print("LLM returned an invalid format, expected a list of strings.")
    except json.JSONDecodeError:
        print("LLM did not return valid JSON. Using empty list for tags.")

    # Return as a list of strings
    print("Final tags list:", tags_list)
    return {"tags": tags_list}
