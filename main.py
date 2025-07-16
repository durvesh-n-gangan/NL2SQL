import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

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
    try:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        result = execute_query_tool.invoke(query_string)
        return result
    except Exception as e:
        error_msg = f"Database Error: {str(e)}"
        return error_msg

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
    
    def execute_and_log_query(inputs):
        query = inputs["query"]
        question = inputs["question"]
        try:
            result = execute_query(query)
            if "Database Error:" in str(result):
                save_query_log(question, query, "ERROR")
            else:
                save_query_log(question, query, "SUCCESS")
            return result
        except Exception as e:
            save_query_log(question, query, f"EXCEPTION: {str(e)}")
            return f"Database Error: {str(e)}"

    parser = StrOutputParser()
    rephrase_answer = RunnableSequence(answer_prompt, llm, parser)
    
    memory_chain = (
        RunnablePassthrough.assign(query=RunnableLambda(generate_query_with_history))
        .assign(result=RunnableLambda(execute_and_log_query))
        | rephrase_answer
    )
    return memory_chain

memory = ConversationBufferWindowMemory(k=3, return_messages=True)

memory_chain = get_memory_chain_with_history(llm, db, memory)

HISTORY_FILE = "question_history.txt"
QUERY_LOG_FILE = "query_log.txt"

def save_question(question):
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(question + "\n")
    except Exception as e:
        print(f"Error saving question: {e}")

def get_saved_questions():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f.readlines() if line.strip()]
                return questions[-10:] if len(questions) > 10 else questions
        return []
    except Exception as e:
        print(f"Error reading questions: {e}")
        return []

def save_query_log(question, query, result_status):
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] Question: {question}\n")
            f.write(f"SQL Query: {query}\n")
            f.write(f"Status: {result_status}\n")
            f.write("-" * 50 + "\n")
    except Exception as e:
        print(f"Error saving query log: {e}")

def handle_query(question):
    try:
        save_question(question)
        
        input_data = {"question": question}
      
        answer = memory_chain.invoke(input_data)
        
        memory.save_context({"input": question}, {"output": answer})
        
        return answer
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}. Please try rephrasing your question."
        memory.save_context({"input": question}, {"output": error_msg})
        return error_msg