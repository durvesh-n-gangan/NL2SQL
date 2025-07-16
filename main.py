import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
os.environ["LANGSMITH_TRACING"] = "true"
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-2.5-pro", model_provider="google_genai")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import Optional
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain.memory import ConversationBufferWindowMemory

class QueryState(BaseModel):
    query: str = Field(description="Make it syntactically correct SQL query only, for the given dialect.")
    top_k: Optional[int] = Field(default=10, description="Number of top results to consider")

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

IMPORTANT MySQL syntax requirements:
- Every derived table (subquery in FROM clause) MUST have an alias
- Use proper MySQL syntax for subqueries
- Example: SELECT count(*) FROM (SELECT DISTINCT customer_id FROM orders) AS customer_counts;

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specified.\n\nIMPORTANT MySQL syntax requirements:\n- Every derived table (subquery in FROM clause) MUST have an alias\n- Use proper MySQL syntax for subqueries\n- Example: SELECT count(*) FROM (SELECT DISTINCT customer_id FROM orders) AS customer_counts;\n\nHere is the relevant table info: {table_info}\n\nConsider the chat history below for context when answering follow up questions."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}"),
])

answer_prompt = PromptTemplate.from_template(
"""You are an assistant that transforms SQL query outputs into natural language answers.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Provide a user-friendly, helpful answer:"""
)

def write_query(state_dict):
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": state_dict.get("top_k", 5),
        "table_info": db.get_table_info(),
        "input": state_dict["question"],
    })
    structured_llm = llm.with_structured_output(QueryState)
    result = structured_llm.invoke(prompt)
    result = dict(result)
    return result['query']

def execute_query(query_string):
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return execute_query_tool.invoke(query_string)

def sync_memory_with_streamlit(memory, chat_history):
    memory.clear()
    for msg in chat_history:
        if msg["role"] == "user":
            memory.save_context({"input": msg["content"]}, {})
        elif msg["role"] == "assistant":
            memory.save_context({}, {"output": msg["content"]})

def get_memory_chain_with_history(llm, db, memory):
    def generate_query_with_history(inputs):
        prompt = memory_prompt.invoke({
            "input": inputs["question"],
            "table_info": db.get_table_info(),
            "messages": memory.chat_memory.messages,
        })
        structured_llm = llm.with_structured_output(QueryState)
        result = structured_llm.invoke(prompt)
        return dict(result)["query"]

    parser = StrOutputParser()
    rephrase_answer = RunnableSequence(answer_prompt, llm, parser)
    
    memory_chain = (
        RunnablePassthrough.assign(query=RunnableLambda(generate_query_with_history))
        .assign(result=RunnableLambda(lambda d: execute_query(d["query"])))
        | rephrase_answer
    )
    return memory_chain

parser = StrOutputParser()
rephrase_answer = RunnableSequence(answer_prompt, llm, parser)
query_chain = RunnablePassthrough.assign(query=RunnableLambda(lambda d: write_query(d))).assign(result=RunnableLambda(lambda d: execute_query(d["query"])))
chain = RunnableSequence(query_chain, rephrase_answer)

# Initialize memory
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Create memory-aware chain
memory_chain = get_memory_chain_with_history(llm, db, memory)

# File to store question history
HISTORY_FILE = "question_history.txt"

def save_question(question):
    """Save a question to the history file"""
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(question + "\n")
    except Exception as e:
        print(f"Error saving question: {e}")

def get_saved_questions():
    """Get list of saved questions from history file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f.readlines() if line.strip()]
                # Return last 10 questions to avoid cluttering the UI
                return questions[-10:] if len(questions) > 10 else questions
        return []
    except Exception as e:
        print(f"Error reading questions: {e}")
        return []

def handle_query(question):
    """Handle a user query and return the answer using conversation memory"""
    try:
        # Save the question to history
        save_question(question)
        
        # Prepare input for memory-aware chain
        input_data = {"question": question}
        
        # Run the memory-aware chain to get the answer
        answer = memory_chain.invoke(input_data)
        
        # Save the conversation to memory
        memory.save_context({"input": question}, {"output": answer})
        
        return answer
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}. Please try rephrasing your question."
        # Still save error conversations to memory for context
        memory.save_context({"input": question}, {"output": error_msg})
        return error_msg

def get_conversation_history():
    """Get the current conversation history from memory"""
    try:
        messages = memory.chat_memory.messages
        history = []
        for msg in messages:
            if hasattr(msg, 'content'):
                msg_type = "Human" if msg.type == "human" else "Assistant"
                history.append(f"{msg_type}: {msg.content}")
        return history
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

def clear_conversation_memory():
    """Clear the conversation memory"""
    try:
        memory.clear()
        return "Conversation memory cleared successfully."
    except Exception as e:
        return f"Error clearing memory: {str(e)}"

# Test the chain (optional - can be removed in production)
if __name__ == "__main__":
    # Test the memory-aware chain
    test_question = "How many customers are there?"
    result = handle_query(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {result}")
    
    # Test follow-up question
    follow_up = "What about orders?"
    result2 = handle_query(follow_up)
    print(f"\nFollow-up: {follow_up}")
    print(f"Answer: {result2}")
    
    # Show conversation history
    print(f"\nConversation History:")
    for msg in get_conversation_history():
        print(f"  {msg}")