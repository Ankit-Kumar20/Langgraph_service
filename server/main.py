from fastapi import FastAPI, Body
from .service import built_graph, HumanMessage, get_user_graph
from typing import TypedDict

app = FastAPI()

@app.get('/')
async def root():
    return({"message": "hello"})

@app.post('/service')
async def service(user_input: dict = Body(...)):
    user_id = user_input['user_id']
    print(user_input['query'])
    config = {"configurable": {
        "thread_id": "symptom_session_1"
    }}

    built_graph = get_user_graph(user_id)

    result = built_graph.invoke({
        "messages": [HumanMessage(content=user_input["query"])]
    }, config=config)
    print("AI: " + result["messages"][-1].content)

    return("AI: " + result["messages"][-1].content)
