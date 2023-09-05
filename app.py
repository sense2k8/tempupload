import streamlit as st
import os
import pdfplumber
import openai
from io import BytesIO
import time
import configparser
import gensim
from gensim import corpora, models, similarities
import glob

# Read configuration values from the config file
config = configparser.ConfigParser()
config.read("config.ini")

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()

#os.environ['REQUESTS_CA_BUNDLE'] = config.get("openai", "REQUESTS_CA_BUNDLE")

# Set up OpenAI API
os.environ["OPENAI_API_KEY"] = config.get("openai", "OPENAI_API_KEY")
openai.api_type = "azure"
openai.api_base = config.get("openai", "api_base")
openai.api_version = "2023-03-15-preview" #2023-05-15
openai.api_key = os.getenv("OPENAI_API_KEY")

# The engine value from the config file
engine = config.get("openai", "engine")

# PDF folder and vector database folder from the config file
pdf_folder = config.get("vector_db", "pdf_folder")
vector_db_folder = config.get("vector_db", "vector_db_folder")


# Add a state variable to control the visibility of the rule book
st.set_page_config(page_title="Demo App",layout="wide")


# st.markdown('''
# <style>
#     [data-testid=stSidebar] {
#         #margin-top: 20px;
#         background-color: #5da6d6;
#         color: #ffffff;
#     }
# </style>
# ''', unsafe_allow_html=True)




def main():
    st.sidebar.title("Gen-AI e-Discovery search craft.")

    
    

    uploaded_file = st.sidebar.file_uploader("Choose a case file", help= "Upload Complain File",type=["pdf"])

    integer_slider = st.sidebar.slider("Search Terms#", 0, 200, 50)

    rank_search_terms = st.sidebar.checkbox("Rank search terms", value=True)
    provide_rationale = st.sidebar.checkbox("Provide rationale", value=True)

    #st.sidebar.markdown('<style>textarea {color: lightgrey;}</style>', unsafe_allow_html=True)
    #special_instructions = st.sidebar.text_area("Special instructions", value="Please provide some composite search terms using the generated search terms")
    
    # Add an Advanced Settings section and a button to show/hide the rule book
    # st.sidebar.subheader("Advanced Settings")
    # show_rule_book = st.sidebar.empty()
    # if show_rule_book.button("Show Rule Book"):
    #     show_rule_book.button("Hide Rule Book")
    #     with open("ruleengine.txt", "r") as file:
    #         rule_book_content = file.read()
    #     st.write(rule_book_content)
    # else:
    #     st.write("")

    if uploaded_file is not None:
        pdf_content = read_pdf(uploaded_file)
        with st.expander("Document Content"):
            st.write(pdf_content)

        if st.button("Generate"):
            with st.spinner("Generating case type..."):
                case_type_keywords = identify_case_type(pdf_content)
            with st.expander("Case Type"):
                st.write(case_type_keywords)

            # with st.spinner("Generating reference document list..."):
            #     document_paths, sorted_sims = search_vector_db(case_type_keywords, vector_db_folder)
            #     reference_documents = ""
            #     for doc_position, doc_score in sorted_sims:
            #         if doc_score > 0.70:
            #             with open(document_paths[doc_position], "rb") as file:
            #                 content = read_pdf(file)                            
            #                 reference_documents += f"\n\n{content}"
            # with st.spinner("Optimizing reference documents..."): 
            #     optimized_reference_content = optimize_reference_documents(case_type_keywords, reference_documents)
            #     with st.expander("Reference Document Stats"):
            #         st.write(f"Reference content Length: {len(reference_documents)}")
            #         st.write(f"Optimized reference content Length: {len(optimized_reference_content)}")
            with st.spinner("Generating search terms..."):                
                #generateSearchTerms(pdf_content, integer_slider, rank_search_terms, provide_rationale, optimized_reference_content)
                generateSearchTerms(pdf_content, integer_slider, rank_search_terms, provide_rationale, "")
    
    # Vector database tab
    # st.sidebar.subheader("Vector Database")
    # if st.sidebar.button("Add files to vector database", disabled=True):
    #     add_files_to_vector_db(pdf_folder, vector_db_folder)
    # query = st.sidebar.text_input("Search vector database")
    # if st.sidebar.button("Test Embeddings"):
    #     search_vector_db(query, vector_db_folder)

def optimize_reference_documents(case_type_keywords, reference_documents):
    optimized_content = ""
    max_length = 30000
    final_length = 5000

    for i in range(0, len(reference_documents), max_length):
        chunk = reference_documents[i:i+max_length]
        #TBD: IMPORTANT                   Need to take from DB authored prompt this is jsut for demo
        cpPrompt = f'''
        As an AI trained in legal matters, please extract only the relevant text related to the following case type keywords from the given reference documents:
        Case Type Keywords: {case_type_keywords}
        Reference Documents:
        {chunk}'''

        chatmemory = []
        chatmemory.append({"role":"system","content": ""})
        chatmemory.append({"role":"user","content": cpPrompt })

        response = openai.ChatCompletion.create(
            engine= engine,
            messages=chatmemory,
            temperature=0,
            max_tokens= final_length,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        optimized_chunk = response.choices[0].message.content
        optimized_content += optimized_chunk

    return optimized_content[:final_length]

def summarize_document(content, summary_ratio):
    cpPrompt = f'''
    As an AI trained in summarization, please summarize the following document to {summary_ratio * 100}% of its original size.
    {content}'''

    chatmemory = []
    chatmemory.append({"role":"system","content": ""})
    chatmemory.append({"role":"user","content": cpPrompt })

    response = openai.ChatCompletion.create(
        engine= engine,
        messages=chatmemory,
        temperature=0,
        max_tokens= 10000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    summarized_content = response.choices[0].message.content
    return summarized_content

def read_pdf(file):
    if hasattr(file, 'getbuffer'):
        data = BytesIO(file.getbuffer())
    else:
        data = file

    with pdfplumber.open(data) as pdf:
        num_pages = len(pdf.pages)
        content = ""
        for page_num in range(num_pages):
            content += pdf.pages[page_num].extract_text() + ""
    return content

def identify_case_type(content):
    cpPrompt = f'''
    As an AI trained in legal matters, please analyze the following complaint and generate up to Max 5 case type keywords.
    {content}'''

    chatmemory = []
    chatmemory.append({"role":"system","content": ""})
    chatmemory.append({"role":"user","content": cpPrompt })

    response = openai.ChatCompletion.create(
        engine= engine,
        messages=chatmemory,
        temperature=0,
        max_tokens= 10000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    case_type_keywords = response.choices[0].message.content
    return case_type_keywords

def generateSearchTerms(content, num_search_terms, rank_search_terms, provide_rationale, reference_documents):
    cpPrompt = f'''
    As an AI trained in legal matters and e-discovery, please analyze the following complaint and generate {num_search_terms} e-discovery search terms. If possible, rank the search terms according to their importance and provide a rationale for each term. 
    Don't repeat the same search term, make a unique list.
    {content}
    Reference Documents:
    {reference_documents}'''

    chatmemory = []
    chatmemory.append({"role":"system","content": ""})
    chatmemory.append({"role":"user","content": cpPrompt })

    res_box = st.empty()
    report = []

    # Add custom CSS for grey background   

    for response in openai.ChatCompletion.create(engine= engine,  messages=chatmemory, temperature=0, max_tokens= 10000, top_p=0.95, frequency_penalty=0,  presence_penalty=0,  stop=None, stream=True ):
           #search_terms = response.choices[0].message.content
           #st.write(search_terms)
        try:
            report.append(str(response.choices[0].delta.content))
            result = "".join(report).strip()
            #result = result.replace("\n", "")        
            res_box.markdown(f'{result}') 
            #res_box.markdown(f'<div class="grey-background">{result}</div>', unsafe_allow_html=True)
            # Add thumbs up and thumbs down buttons
           
            
            
        except:
            pass

    # api_usage = f'''
    # **API Usage Insights:**
    # - Total tokens used in request: {response.usage["total_tokens"]}
    # - Total tokens used in response: {response.usage["prompt_tokens"]}    
    # - Approximate cost: ${(response.usage["total_tokens"] / 1000) * 0.06:.2f}
    # '''
    # st.markdown(api_usage, unsafe_allow_html=True)

def add_files_to_vector_db(pdf_folder, vector_db_folder):
    # Read all PDF files in the folder
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    # Extract text from PDF files
    documents = []
    document_paths = []  # Store document paths
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as file:
            content = read_pdf(file)
            documents.append(content)
            document_paths.append(pdf_file)  # Add document path to the list

    # Save document paths
    with open(os.path.join(vector_db_folder, "document_paths.txt"), "w") as file:
        for path in document_paths:
            file.write(f"{path}")

    # Create a vector datastore using Gensim
    texts = [[word for word in document.lower().split()] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    index = similarities.MatrixSimilarity(lsi[corpus])

    # Save the vector datastore
    dictionary.save(os.path.join(vector_db_folder, "dictionary.dict"))
    corpora.MmCorpus.serialize(os.path.join(vector_db_folder, "corpus.mm"), corpus)
    lsi.save(os.path.join(vector_db_folder, "lsi.lsi"))
    index.save(os.path.join(vector_db_folder, "index.index"))

    st.write("Files added to vector database successfully.")

def search_vector_db(query, vector_db_folder):
    # Load the vector datastore
    dictionary = corpora.Dictionary.load(os.path.join(vector_db_folder, "dictionary.dict"))
    corpus = corpora.MmCorpus(os.path.join(vector_db_folder, "corpus.mm"))
    lsi = models.LsiModel.load(os.path.join(vector_db_folder, "lsi.lsi"))
    index = similarities.MatrixSimilarity.load(os.path.join(vector_db_folder, "index.index"))

    # Load document paths
    with open(os.path.join(vector_db_folder, "document_paths.txt"), "r") as file:
        document_paths = [line.strip() for line in file.readlines()]

    # Query the vector datastore
    query_bow = dictionary.doc2bow(query.lower().split())
    query_lsi = lsi[query_bow]
    sims = index[query_lsi]

    # Show the results
    sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    with st.expander("Reference Documents"):
        for doc_position, doc_score in sorted_sims:
            st.write(f"Document Index: {doc_position} Document path: {document_paths[doc_position]} Confidence : {doc_score:.2f}")
            #st.write(f"Document path: {document_paths[doc_position]}")  # Display document path
    return document_paths, sorted_sims

# Custom CSS to change the sidebar color


if __name__ == "__main__":
    main()