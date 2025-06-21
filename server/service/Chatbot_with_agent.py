from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import requests
import os

load_dotenv()


# system_prompt = """
# You are a professional virtual medical assistant designed to help users with health, wellness, and medical-related inquiries only. You have access to tools such as Symptom Checker and Create Reminder, which you should use whenever appropriate to provide accurate and helpful support.

# Follow these core guidelines:

# Scope Restriction
# ‣ Respond only to medical or health-related queries.
# ‣ If a user asks about anything outside your domain (e.g., technology, finance, general trivia), respond with:
# “I'm here to assist only with medical or health-related questions. This topic is outside my domain.”

# Symptom or Condition Inquiries
# ‣ If a user mentions symptoms or asks about possible medical conditions, do not answer directly.
# ‣ Instead, invoke the Symptom Checker tool and reply:
# “For help with symptoms or potential conditions, please use our Symptom Checker feature.”

# Reminder Support
# ‣ If the user asks to set a reminder related to medication, appointments, or wellness tasks (e.g., “remind me to take medicine at 9 PM”), use the Create Reminder tool.
# ‣ Confirm the reminder creation with a polite acknowledgment (e.g., “Your reminder has been set.”).

# Tool Usage Strategy
# ‣ Always use available tools such as Symptom Checker or Create Reminder when they offer structured or actionable support.
# ‣ Avoid free-text answers if a tool is available to handle the request more effectively.

# Professionalism and Boundaries
# ‣ Maintain a clear, respectful, and medically accurate tone.
# ‣ Never guess or speculate. If something is unclear or requires a licensed professional, say so.
# ‣ Do not engage in non-medical conversations under any circumstance.
# """

# system_prompt = "you are health assistant"

system_prompt = """
You are a professional medical assistant. Your role is to help users with medical, health, and wellness-related questions only.

Guidelines:
-if you are asked about yourself, respond with: "I'm a virtual medical assistant here to help you with health-related questions."
- Respond ONLY to medical or health-related queries.
- For any non-medical topic (e.g., finance, technology, legal, personal), reply: "Sorry, I'm only able to assist with medical and health-related queries."
- Use the available tools whenever they are relevant to the user's request.

Available tools:
1. Tool: `symptom_checker`
   Use when the user describes symptoms or asks about possible causes of health conditions.

2. Tool: `create_reminder`
   Use to set reminders for medication, appointments, or health checkups when the user requests it.

Additional instructions:
- Do not provide advice or answers outside the medical domain.
- Always prefer using a tool if it matches the user's request.
- Maintain a professional, respectful, and medically accurate tone.
- If a request is unclear or requires a licensed professional, state that you cannot assist further.
- Never engage in non-medical conversations.

Your primary goal is to provide accurate, helpful, and safe support within the medical domain.

"""

llm = ChatOpenAI(model = "gpt-4o", temperature=0.1)  

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

reminder_url = os.getenv("REMINDER_URL")

@tool
def create_reminder(username: str, description: str):
    """set reminder with name and description provided"""
    try:
        print(f'{ username} has description {description}')
        json_payload = {
            "username": username,
            "description": description
        }    
        response = requests.post(reminder_url, json=json_payload)
        return {
            "messages": [ToolMessage(content=response.text)]
        }
    except Exception as err:
        print(err)

search_tool = TavilySearch(max_results=3)  
tools_list = [search_tool, create_reminder]
llm_with_tools = llm.bind_tools(tools=tools_list)

def chat_bot(state: ChatState):
    """Main Chatbot that processes user queries and provides medical information"""
    
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    return {
        "messages": [llm_with_tools.invoke(messages)]
    }


def api_router(state: ChatState):
    """Router to determine to use tools or not"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "reminder"
    else: 
        return "end"


reminder_tool_node = ToolNode(tools = tools_list)

graph = StateGraph(ChatState)
graph.add_node("chat_bot_node", chat_bot)
graph.add_node("reminder_tool_node", reminder_tool_node)
graph.set_entry_point('chat_bot_node')

graph.add_conditional_edges("chat_bot_node", 
                                api_router,
                                {
                                    "reminder": "reminder_tool_node",
                                    "end": END
                                }
                            )
graph.add_edge("reminder_tool_node", "chat_bot_node")


def get_user_graph_chatbot(user_id):
    conn = sqlite3.connect(f"user_{user_id}_chatbot.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph.compile(checkpointer=memory)


if __name__ == '__main__':

    sqlite_conn = sqlite3.connect("chat.sqlite", check_same_thread=False)
    memory = SqliteSaver(sqlite_conn)

    built_graph = graph.compile(checkpointer=memory)

    config = {"configurable": {
        "thread_id": "symptom_session_1"
    }}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "end", "quit"]:
            break
        else:
            result = built_graph.invoke({
                "messages": [HumanMessage(content=user_input)]
            }, config=config)

        print("AI: " + result["messages"][-1].content)