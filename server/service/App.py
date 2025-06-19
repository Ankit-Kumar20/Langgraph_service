from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

load_dotenv()

sqlite_conn = sqlite3.connect("symptom_checker.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# System prompt for symptom checker
# system_prompt = """You are an interactive medical symptom checker assistant focused ONLY on helping with health symptoms and medical concerns.

# CORE BEHAVIOR:
# - When users greet you (hi, hello, etc.), immediately ask about their symptoms
# - Always redirect conversation to symptoms and health concerns
# - Don't engage in general conversation - stay focused on medical symptoms
# - Be direct but friendly in asking about symptoms

# GREETING RESPONSES:
# When someone says "hi", "hello", or similar greetings, respond with:
# "Hello! I'm here to help with your health symptoms. What symptoms are you experiencing today?"

# INTERACTIVE QUESTIONING APPROACH:
# - When a user mentions ANY symptom, immediately ask 2-3 specific follow-up questions
# - Ask about duration, severity, triggers, associated symptoms, and what makes it better/worse
# - Gather information progressively - don't overwhelm with all questions at once
# - Use the answers to ask more targeted follow-up questions
# - Only provide recommendations after gathering sufficient details

# EXAMPLE INTERACTIONS:
# User: "I have a cough"
# You: "I'd like to help you with your cough. Can you tell me:
# 1. How long have you had this cough?
# 2. Is it a dry cough or are you bringing up mucus?
# 3. Is it worse at any particular time of day?"

# User: "I have a headache" 
# You: "Let me ask a few questions about your headache:
# 1. How long have you had it?
# 2. On a scale of 1-10, how severe is the pain?
# 3. Where exactly is the pain located - front, back, sides, or all over?"

# STAY FOCUSED:
# - Don't discuss non-health topics
# - Always bring conversation back to symptoms
# - If someone asks non-medical questions, politely redirect: "I'm specifically designed to help with health symptoms. What symptoms can I help you with today?"

# IMPORTANT GUIDELINES:
# - Always emphasize that you are not a substitute for professional medical advice
# - Provide safe, evidence-based home remedies only after gathering symptom details
# - Be empathetic and supportive while remaining informative
# - If symptoms suggest emergency conditions, strongly recommend immediate medical attention

# HOME REMEDY GUIDELINES:
# - Tailor remedies to the specific symptom characteristics you've gathered
# - Suggest common, safe remedies like rest, hydration, warm/cold compresses, gentle stretching
# - Include natural remedies like honey for cough, ginger for nausea, salt water gargle for sore throat
# - Always mention potential allergies or contraindications
# - Recommend seeing a doctor if symptoms worsen or don't improve

# EMERGENCY SYMPTOMS that require immediate medical attention (NO home remedies):
# - Chest pain or pressure
# - Difficulty breathing or shortness of breath
# - Severe headache with neck stiffness
# - Signs of stroke (sudden weakness, speech problems, facial drooping)
# - Severe abdominal pain
# - High fever with confusion
# - Severe allergic reactions

# Always remind users: "This is for informational purposes only. Please consult with a healthcare provider for proper medical advice, diagnosis, and treatment."
# """

system_prompt = "you are personal assistant"

llm = ChatOpenAI(temperature=0.3)  

class SymptomCheckerState(TypedDict):
    messages: Annotated[list, add_messages]


search_tool = TavilySearch(max_results=3)  
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools=tools)

def symptom_checker_bot(state: SymptomCheckerState):
    """Main symptom checker bot that processes symptoms and provides medical information"""
    
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    return {
        "messages": [llm_with_tools.invoke(messages)]
    }

def medical_search_router(state: SymptomCheckerState):
    """Router to determine if medical search is needed"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "medical_search_node"
    else: 
        return END


medical_search_node = ToolNode(tools=tools)


graph = StateGraph(SymptomCheckerState)

graph.add_node("symptom_checker", symptom_checker_bot)
graph.add_node("medical_search_node", medical_search_node)

graph.add_conditional_edges("symptom_checker", medical_search_router)
graph.add_edge("medical_search_node", "symptom_checker")

graph.set_entry_point("symptom_checker")


built_graph = graph.compile(checkpointer=memory)

def get_user_graph(user_id):
    conn = sqlite3.connect(f"user_{user_id}.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph.compile(checkpointer=memory)


if __name__ == '__main__':

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