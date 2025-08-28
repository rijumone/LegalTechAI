import os
import json
import uuid
from tempfile import NamedTemporaryFile
import pandas as pd
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from subprocess import run
from loguru import logger
from typing import Annotated
from markitdown import MarkItDown
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.id import ID
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient, models
# from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import VectorParams, Distance, PointStruct

_appendix = """
Rhetorical Roles Definititions
| Rhetorical Role                          | Rhetorical Roles (sentence level)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Preamble<br>(PREAMBLE)                   | A typical judgement would start with the court name, the details of parties, lawyers and judges' names, Headnotes. This section typically would end with a keyword like (JUDGEMENT or ORDER etc.)<br>Some supreme court cases also have HEADNOTES, ACTS section. They are also part of Preamble.                                                                                                                                                                                                                                                                                                               |
| Facts(FAC)                               | This refers to the chronology of events (but not judgement by lower court) that led to filing the case, and how the case evolved over time in the legal system (e.g., First Information Report at a police station, filing an appeal to the Magistrate, etc.)<br>Depositions and proceedings of current court<br>Summary of lower court proceedings                                                                                                                                                                                                                                                            |
| Ruling by Lower Court (RLC)              | Judgments given by the lower courts (Trial Court, High Court) based on which the present appeal was made (to the Supreme Court or high court). The verdict of the lower Court, Analysis & the ratio behind the judgement by the lower Court is annotated with this label.                                                                                                                                                                                                                                                                                                                                      |
| Issues (ISSUE)                           | Some judgements mention the key points on which the verdict needs to be delivered. Such Legal Questions Framed by the Court are ISSUES.<br>E.g. “he point emerge for determination is as follow:- (i) Whether on 06.08.2017 the accused persons in furtherance of their common intention intentionally caused the death of the deceased by assaulting him by means of axe ?”                                                                                                                                                                                                                                   |
| Argument by Petitioner (ARG_PETITIONER) | Arguments by petitioners' lawyers. Precedent cases argued by petitioner lawyers fall under this but when court discusses them later then they belong to either the relied / not relied upon category.<br>E.g. “learned counsel for petitioner argued that …”                                                                                                                                                                                                                                                                                                                                                   |
| Argument by Respondent (ARG_RESPONDENT) | Arguments by respondents lawyers. Precedent cases argued by respondent lawyers fall under this but when court discusses them later then they belong to either the relied / not relied upon category.<br>E.g. “learned counsel for the respondent argued that …”                                                                                                                                                                                                                                                                                                                                                |
| Analysis (ANALYSIS)                      | Courts discussion on the evidence,facts presented,prior cases and statutes. These are views of the court. Discussions on how the law is applicable or not applicable to current case. Observations(non binding) from court. It is the parent tag for 3 tags: PRE_RLEIED, PRE_NOT_RELIED and STATUTE i.e. Every statement which belong to these 3 tags should also be marked as ANALYSIS<br><br>E.g. “Post Mortem Report establishes that .. “<br>E.g. “In view of the abovementioned findings, it is evident that the ingredients of Section 307 have been made out ….”                                     |
| Statute (STA)                            | Text in which the court discusses Established laws, which can come from a mixture of sources – Acts , Sections, Articles, Rules, Order, Notices, Notifications, Quotations directly from the bare act, and so on.<br>Statute will have both the tags Analysis + Statute<br><br>E.g. “Court had referred to Section 4 of the Code, which reads as under: "4. Trial of offences under the Indian Penal Code and other laws.-- (1) All offences under the Indian Penal Code (45 of 1860) shall be investigated, inquired into, tried, and otherwise dealt with according to the provisions hereinafter contained” |
| Precedent Relied (PRE_RELIED)           | Sentences in which the court discusses prior case documents, discussions and decisions which were relied upon by the court for final decisions.<br>So Precedent will have both the tags Analysis + Precedent<br>E.g. This Court in Jage Ram v. State of Haryana3 held that: "12. For the purpose of conviction under Section 307 IPC, ….. “                                                                                                                                                                                                                                                                    |
| Precedent Not Relied (PRE_NOT_RELIED)  | Sentences in which the court discusses prior case documents, discussions and decisions which were not relied upon by the court for final decisions. It could be due to the fact that the situation in that case is not relevant to the current case.<br>E.g. This Court in Jage Ram v. State of Haryana3 held that: "12. For the purpose of conviction under Section 307 IPC, ….. “                                                                                                                                                                                                                            |
| Ratio of the decision (Ratio)            | Main Reason given for the application of any legal principle to the legal issue. This is the result of the analysis by the court.<br>This typically appears right before the final decision.<br>This is not the same as “Ratio Decidendi” taught in the Legal Academic curriculum.<br>E.g. “The finding that the sister concern is eligible for more deduction under Section 80HHC of the Act is based on mere surmise and conjectures also does not arise for consideration.”                                                                                                                                 |
| Ruling by Present Court (RPC)            | Final decision + conclusion + order of the Court following from the natural / logical outcome of the rationale<br>E.g. “In the result, we do not find any merit in this appeal. The same fails and is hereby dismissed.”                                                                                                                                                                                                                                                                                                                                                                                       |
| NONE                                     | If a sentence does not belong to any of the above categories<br>E.g. “We have considered the submissions made by learned counsel for the parties and have perused the record.”                                                                                                                                                                                                                                                                                                                                                                                                                                 |
"""

client = Client()
qd_client = QdrantClient(host="localhost", port=6333)
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

(client
  .set_endpoint(os.getenv('APPWRITE_ENDPOINT')) # Your API Endpoint
  .set_project(os.getenv('APPWRITE_PROJECT')) # Your project ID
  .set_key(os.getenv('APPWRITE_API_KEY')) # Your secret API key
  .set_self_signed() # Use only on dev mode with a self-signed SSL cert
)

databases = Databases(client)

def get_ollama_llm(
        model_name: str,
        temperature: float = 0.8,
    ):
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=os.getenv('OLLAMA_BASE_URL'),
        client_kwargs={
            'headers': {
                'Authorization': f'Bearer {os.getenv("OLLAMA_API_KEY")}',
                'Content-Type': 'application/json',
            },
        }
    )
    return llm

def write_to_appwrite(collection_id, appwrite_msg: dict):
    return databases.create_document(
        database_id=os.getenv('APPWRITE_DATABASE'),
        collection_id=collection_id,
        document_id=ID.unique(),
        data={**appwrite_msg}
    )

def fetch_messages(thread_id: str):
    """Fetches messages for a given thread ID from Appwrite database."""
    try:
        documents = databases.list_documents(
            database_id=os.getenv('APPWRITE_DATABASE'),
            collection_id=os.getenv('APPWRITE_MESSAGES_COLLECTION'),
            queries=[json.dumps({"method":"equal","attribute":"thread","values":[thread_id]})]
        )
        return documents['documents']
    except Exception as e:
        print(f"Error fetching messages for thread {thread_id}: {e}")
        return []

def get_markdown_from_pdf(pdf_file_path):
    md = MarkItDown()
    result = md.convert(pdf_file_path)
    return result.text_content

def ask_llm(llm, query):
    response = llm.stream(f'{query}')
    return response

def rm_pdf_4m_sess():
    for key in st.session_state.keys():
        if key in st.session_state:
            del st.session_state[key]


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_message: str
    ai_message: str
    thread_id: str
    ai_location: str
    ai_activity: str
    is_image: bool
    rhetorical_role: str | None
    retrieved_docs: list[Document] | None


system_prompt = ''
system_prompt += f"""You are a helpful AI Legal Assistant.\n"""


def _separate_reasoning_from_response(response: str):
    """Given a str response, if it starts with <think>, extract and return the text inside <think> and </think> tags."""
    if response.startswith("<think>"):
        start = response.find("<think>") + len("<think>")
        end = response.find("</think>")
        if end != -1:
            return response[start:end].strip(), response[end+8:].strip()  # Exclude <think> and </think> tags
    return None, response

def log_message_to_appwrite(state: ChatState):
    thread_id = state["thread_id"]
    appwrite_msg = {
        "role": "user",
        "content": state.get("user_message"),
        "thread": thread_id,
    }
    # Save user message to Appwrite database
    appwrite_result = write_to_appwrite(os.getenv('APPWRITE_MESSAGES_COLLECTION'), appwrite_msg)
    messages = fetch_messages(thread_id)
    state["messages"] = [{"role": _m['role'], "content": _m['content']} for _m in messages]
    # import pdb; pdb.set_trace()
    return state

def infer_rhetorical_role(state: ChatState):
    """Infer rhetorical roles from the user message."""
    _system_prompt = system_prompt
    _system_prompt += "\n\nYou are an AI Legal Assistant that can infer rhetorical roles from legal documents.\n"
    _system_prompt += "\n\nGiven below are the possible rhetoric roles, choose ONLY the MOST SUITABLE ONE.\n"
    _system_prompt += "\n\nOutput should only be the role.\n"
    _system_prompt += f"\n\n{_appendix}\n\n"
    context = [{"role": "system", "content": _system_prompt}] + [{"role": "user", "content": state.get("user_message")}]
    agent = get_ollama_llm(model_name=st.session_state.selected_model)
    # Stream the response
    response = ""
    for chunk in agent.stream(context):
        # import pdb;pdb.set_trace()
        print(chunk.content, end="", flush=True)
        response += chunk.content
    raw_rr = response.strip()
    allowed_responses = ["PREAMBLE", "FAC", "RLC", "ISSUE", "ARG_PETITIONER", "ARG_RESPONDENT",
                         "ANALYSIS", "STA", "PRE_RELIED", "PRE_NOT_RELIED", "Ratio", "RPC", "NONE"]
    for _r in allowed_responses:
        if _r in raw_rr:
            raw_rr = _r
            break
    if raw_rr not in allowed_responses:
        # Extract the rhetorical role from the response
        context = [{"role": "system", "content": 
                    f"""Which of these is the input most likely to be:\n 
                    {allowed_responses}\n
                    OUTPUT should be only the role name, no other text.""",    
                }] + [{"role": "user", "content": raw_rr}]
        logger.warning(context)
        state['rhetorical_role'] = agent.invoke(context).content
    else:
        state['rhetorical_role'] = raw_rr
    logger.warning(f"Rhetorical Role Inference: {state['rhetorical_role']}")
    return state

def retrieve_context_from_qdrant(state: ChatState):
    query_embedding = embedder.embed_query(state.get("user_message"))
    search_results = qd_client.search(
        collection_name=st.session_state.thread_id,
        query_vector=query_embedding,
        limit=10,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="labels",
                    match=models.MatchValue(value=state['rhetorical_role'])
                ),
            ]
        ),
    )
    state['retrieved_docs'] = [
        Document(
            page_content=point.payload.get("text", ""),
            labels=point.payload.get("labels", {}),
        )
        for point in search_results
        if point.payload is not None
    ]
    logger.warning(state['retrieved_docs'])
    return state


def chatbot_node(state: ChatState):
    n_memory = state.get("n_memory", 2)
    user_message = state.get("user_message")
    thread_id = state.get("thread_id")
    if not user_message:
        return state
    # Maintain chat memory
    messages = state.get("messages", [])
    # _msg = {"role": "user", "content": user_message}
    # messages.append(_msg)
    # Prepare context: system prompt + last n messages
    _system_prompt = system_prompt
    _system_prompt += "\n Retrieved docs: {} \n".format(", ".join([doc.page_content for doc in state.get("retrieved_docs", [])]))
    # logger.warning(_system_prompt)
    # import pdb;pdb.set_trace()
    context = [{"role": "system", "content": _system_prompt}] + messages[-n_memory:]
    # Connect to OllamaChat Agent with custom URL
    agent = get_ollama_llm(model_name=st.session_state.selected_model)
    # Stream the response
    response = ""
    for chunk in agent.stream(context):
        # import pdb;pdb.set_trace()
        print(chunk.content, end="", flush=True)
        response += chunk.content
    reasoning, content = _separate_reasoning_from_response(response)
    state['ai_message'] = content.strip()
    _msg = {"role": "ai", "content": content,}
    appwrite_msg = _msg.copy()
    # appwrite_msg['reasoning'] = reasoning
    appwrite_msg['thread'] = thread_id
    appwrite_result = write_to_appwrite(os.getenv('APPWRITE_MESSAGES_COLLECTION'), appwrite_msg)
    messages = fetch_messages(thread_id)
    state["messages"] = [{"role": _m['role'], "content": _m['content']} for _m in messages]
    state["ai_message"] = response
    return state


# Build the graph
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.add_node("log_message_to_appwrite", log_message_to_appwrite)
graph.add_node("infer_rhetorical_role", infer_rhetorical_role)
graph.add_node("retrieve_context_from_qdrant", retrieve_context_from_qdrant)

graph.add_edge(START, "log_message_to_appwrite")
graph.add_edge("log_message_to_appwrite", "infer_rhetorical_role")
graph.add_edge("infer_rhetorical_role", "retrieve_context_from_qdrant")
graph.add_edge("retrieve_context_from_qdrant", "chatbot")
graph.add_edge("chatbot", END)

workflow = graph.compile()




def main():
    print(workflow.get_graph().print_ascii())
    st.set_page_config(page_title=os.getenv("APP_NAME"), layout="wide")
    st.title(os.getenv("APP_NAME"))
    # chat_panel, info_panel = st.columns(2)
    if not 'thread_id' in st.session_state:
        # create new thread in appwrite
        _thr = {'name': f'thr-{str(uuid.uuid4())}', }
        thread_id = write_to_appwrite(os.getenv('APPWRITE_THREADS_COLLECTION'), _thr).get('$id')
        st.session_state.thread_id = thread_id

    # with chat_panel:
    def chat_callback():
        current_state = {"user_message": None, "n_memory": 2, "thread_id": st.session_state.thread_id}
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        user_input = st.session_state.user_input
        logger.info(f'user_input: {user_input}')
        current_state["user_message"] = user_input
        for event in workflow.stream(current_state):
            for value in event.values():
                # if value is not None:
                    # try:
                    #     print("Astrid:", value["messages"][-1].content)
                    # except AttributeError:
                    #     print("Astrid:", value["messages"][-1]['content'])
                    
                messages = fetch_messages(st.session_state.thread_id)
                st.session_state.messages = [{"role": _m['role'], "content": _m['content']} for _m in messages]
                logger.debug(f"state messages: {st.session_state.messages}")
                current_state = value
        

    # Add model selection dropdown
    available_models = [
        "qwen2.5:7b", 
        "llama3.1:latest", 
        "llama3.2:3b", 
        "phi4:latest", 
        "gemma2:latest",
    ]
    selected_model = st.selectbox("Select AI Model", available_models, index=0)
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = selected_model
        
        
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", on_change=rm_pdf_4m_sess)

    if uploaded_file is None:
        return
    # Read the PDF file
    if 'pdf_file_b' not in st.session_state:
        st.session_state.pdf_file_b = uploaded_file.getvalue()
    pdf_viewer(st.session_state.pdf_file_b, height=900)
    # import pdb;pdb.set_trace()
    if not qd_client.collection_exists(st.session_state.thread_id):
        with NamedTemporaryFile(prefix=f'{st.session_state.thread_id}_', delete=False) as _tf:
            _tf.write(uploaded_file.getvalue())

            if 'mkdwn_4m_pdf' not in st.session_state:
                logger.debug(f'{_tf.name}')
                st.session_state.mkdwn_4m_pdf = get_markdown_from_pdf(_tf.name)
                logger.info(f'mkdwn_4m_pdf: {st.session_state.mkdwn_4m_pdf[:100]}...')

        with st.spinner("Processing PDF text to inference data type..."):
            # Preparing raw data
            _raw_data = [{"id": 1, "data": { "text": st.session_state.mkdwn_4m_pdf } }]
            with NamedTemporaryFile(prefix=f'{st.session_state.thread_id}_raw_', suffix='.json', delete=False) as _rtf:
                _rtf.write(json.dumps(_raw_data).encode())

            
            _custom_proc_in_f = f"{_tf.name}_proc.json"
            _cmd = f'/home/admin/Kitchen/LegalTechAI/rhetorical-role-baseline/.venv/bin/python /home/admin/Kitchen/LegalTechAI/rhetorical-role-baseline/infer_data_prep.py {_rtf.name} {_custom_proc_in_f}'
            run(_cmd.split(), check=True)
            logger.info(f"Custom processed input file: {_custom_proc_in_f}")
            st.write("Custom processed input file:", _custom_proc_in_f)

        with st.spinner("Running inference on processed data..."):
            with NamedTemporaryFile(prefix=f'{st.session_state.thread_id}_out_', suffix='.json', delete=False) as _otf:
                # _otf.write(json.dumps(_raw_data).encode())
                _cmd = f'/home/admin/Kitchen/LegalTechAI/rhetorical-role-baseline/.venv/bin/python /home/admin/Kitchen/LegalTechAI/rhetorical-role-baseline/infer_new.py {_custom_proc_in_f} {_otf.name} /home/admin/Kitchen/LegalTechAI/rhetorical-role-baseline/model.pt'
                run(_cmd.split(), check=True)
                logger.info(f"Inference output file: {_otf.name}")
                st.write("Inference output file:", _otf.name)

        
        
        # Display the processed dataframe
        # --- Step 1: Load JSON file ---
        with open(_otf.name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- Step 2: Extract all results ---
        results = []
        for item in data:
            for ann in item.get("annotations", []):
                for res in ann.get("result", []):
                    results.append(res["value"])  # only take the 'value' dict

        with st.spinner("Generating embeddings..."):
            embeddings = embedder.embed_documents([_['text'] for _ in results])
            st.write(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}")

        # Create a new collection in Qdrant and load the data as vector points
        with st.spinner("Loading data to Qdrant..."):

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload={"text": results[i]['text'], "labels": results[i]['labels']}
                )
                for i in range(len(results))
            ]
            qd_client.create_collection(
                collection_name=st.session_state.thread_id,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
            )
            qd_client.upload_points(collection_name=st.session_state.thread_id, points=points)
            st.success(f"Uploaded {len(points)} chunks to Qdrant collection '{st.session_state.thread_id}'!")

        # --- Step 3: Convert to DataFrame ---
        st.session_state.inferred_df = pd.DataFrame(results, columns=["labels", "text",])

    st.dataframe(st.session_state.inferred_df, use_container_width=True)

    st.chat_input(
        "Ask a question about the paper",
        on_submit=chat_callback,
        key='user_input',
    )
    if 'messages' not in st.session_state:
        return
    
    for idx, message in enumerate(st.session_state.messages):
        logger.debug(message)
        with st.chat_message(message["role"]):
            # Handle rendering of generator, even tho Streamlit handles it automatically
            # but the string needs to be saved back to the message for continuity
            msg = message["content"]
            # check if msg is a generator
            if isinstance(msg, str):
                st.write(msg)
            else:
                ai_res_plchldr = st.empty()
                ai_response = ""
                for chunk in msg:
                    ai_response += chunk.content  # Append each chunk to the response text
                    ai_res_plchldr.write(ai_response)
                st.session_state.messages[idx]["content"] = ai_response



if __name__ == "__main__":
    main()