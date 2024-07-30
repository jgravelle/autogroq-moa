import streamlit as st
import json
from langchain_groq import ChatGroq
from typing import Iterable
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk
from streamlit_ace import st_ace
import copy

if 'query' not in st.session_state:
    st.session_state.query = ""

# Default configuration
default_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 3,
    "layer_agent_config": {}
}

layer_agent_config_def = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama3-8b-8192"
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "gemma-7b-it",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama3-8b-8192"
    },

}

# Recommended Configuration

rec_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 2,
    "layer_agent_config": {}
}

layer_agent_config_rec = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama3-8b-8192",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "llama3-8b-8192",
        "temperature": 0.2
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama3-8b-8192",
        "temperature": 0.4
    },
    "layer_agent_4": {
        "system_prompt": "You are an expert planner agent. Create a plan for how to answer the human's query. {helper_response}",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.5
    },
}

def update_query():
    st.session_state.query = st.session_state.chat_input

def get_rephrased_user_prompt(user_request):
    return f"""Act as a professional prompt engineer and refactor the following 
                user request into an optimized prompt. This agent's goal is to rephrase the request 
                with a focus on the satisfying all following the criteria without explicitly stating them:
        1. Clarity: Ensure the prompt is clear and unambiguous.
        2. Specific Instructions: Provide detailed steps or guidelines.
        3. Context: Include necessary background information.
        4. Structure: Organize the prompt logically.
        5. Language: Use concise and precise language.
        6. Examples: Offer examples to illustrate the desired output.
        7. Constraints: Define any limits or guidelines.
        8. Engagement: Make the prompt engaging and interesting.
        9. Feedback Mechanism: Suggest a way to improve or iterate on the response.
        Apply introspection and reasoning to reconsider your own prompt[s] to:
        Clarify ambiguities
        Break down complex tasks
        Provide essential context
        Structure logically
        Use precise, concise language
        Include relevant examples
        Specify constraints
        Do NOT reply with a direct response to these instructions OR the original user request. Instead, rephrase the user's request as a well-structured prompt, and
        return ONLY that rephrased prompt. Do not preface the rephrased prompt with any other text or superfluous narrative.
        Do not enclose the rephrased prompt in quotes. This agent will be successful only if it returns a well-formed rephrased prompt ready for submission as an LLM request.
        User request: "{user_request}"
        Rephrased:
    """

def rephrase_prompt(user_request):
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
    rephrased_prompt = llm.invoke(get_rephrased_user_prompt(user_request))
    return rephrased_prompt.content


def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message['delta'])
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, output in enumerate(outputs):
                    with cols[i]:
                        st.expander(label=f"Agent {i+1}", expanded=False).write(output)
            
            # Clear layer outputs for the next iteration
            layer_outputs = {}
            
            # Yield the main agent's output
            yield message['delta']

def set_moa_agent(
    main_model: str = default_config['main_model'],
    cycles: int = default_config['cycles'],
    layer_agent_config: dict[dict[str, any]] = copy.deepcopy(layer_agent_config_def),
    main_model_temperature: float = 0.1,
    override: bool = False
):
    if override or ("main_model" not in st.session_state):
        st.session_state.main_model = main_model
    else:
        if "main_model" not in st.session_state: st.session_state.main_model = main_model 

    if override or ("cycles" not in st.session_state):
        st.session_state.cycles = cycles
    else:
        if "cycles" not in st.session_state: st.session_state.cycles = cycles

    if override or ("layer_agent_config" not in st.session_state):
        st.session_state.layer_agent_config = layer_agent_config
    else:
        if "layer_agent_config" not in st.session_state: st.session_state.layer_agent_config = layer_agent_config

    if override or ("main_temp" not in st.session_state):
        st.session_state.main_temp = main_model_temperature
    else:
        if "main_temp" not in st.session_state: st.session_state.main_temp = main_model_temperature

    cls_ly_conf = copy.deepcopy(st.session_state.layer_agent_config)
    
    if override or ("moa_agent" not in st.session_state):
        st.session_state.moa_agent = MOAgent.from_config(
            main_model=st.session_state.main_model,
            cycles=st.session_state.cycles,
            layer_agent_config=cls_ly_conf,
            temperature=st.session_state.main_temp
        )

    del cls_ly_conf
    del layer_agent_config

st.set_page_config(
    page_title="Mixture-Of-Agents Powered by Groq",
    page_icon='static/favicon.ico',
        menu_items={
        'About': "## Groq Mixture-Of-Agents \n Powered by [Groq](https://groq.com)"
    },
    layout="wide"
)
valid_model_names = [
    'llama3-70b-8192',
    'llama3-8b-8192',
    'gemma-7b-it',
    'gemma2-9b-it',
    'mixtral-8x7b-32768'
]

st.markdown("<a href='https://groq.com'><img src='app/static/banner.png' width='500'></a>", unsafe_allow_html=True)
st.write("---")



# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

set_moa_agent()

# Sidebar for configuration
with st.sidebar:
    # config_form = st.form("Agent Configuration", border=False)
    st.title("MOA Configuration")
    with st.form("Agent Configuration", border=False):
        if st.form_submit_button("Use Recommended Config"):
            try:
                set_moa_agent(
                    main_model=rec_config['main_model'],
                    cycles=rec_config['cycles'],
                    layer_agent_config=layer_agent_config_rec,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")
        # Main model selection
        new_main_model = st.selectbox(
            "Select Main Model",
            options=valid_model_names,
            index=valid_model_names.index(st.session_state.main_model)
        )

        # Cycles input
        new_cycles = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=10,
            value=st.session_state.cycles
        )

        # Main Model Temperature
        main_temperature = st.number_input(
            label="Main Model Temperature",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        # Layer agent configuration
        tooltip = "Agents in the layer agent configuration run in parallel _per cycle_. Each layer agent supports all initialization parameters of [Langchain's ChatGroq](https://api.python.langchain.com/en/latest/chat_models/langchain_groq.chat_models.ChatGroq.html) class as valid dictionary fields."
        st.markdown("Layer Agent Config", help=tooltip)
        new_layer_agent_config = st_ace(
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            language='json',
            placeholder="Layer Agent Configuration (JSON)",
            show_gutter=False,
            wrap=True,
            auto_update=True
        )

        if st.form_submit_button("Update Configuration"):
            try:
                new_layer_config = json.loads(new_layer_agent_config)
                set_moa_agent(
                    main_model=new_main_model,
                    cycles=new_cycles,
                    layer_agent_config=new_layer_config,
                    main_model_temperature=main_temperature,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

    st.markdown("---")
    st.markdown("""
    ### Credits
    - MOA: [Together AI](https://www.together.ai/blog/together-moa)
    - LLMs: [Groq](https://groq.com/)
    - Paper: [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
    """)

# Main app layout
st.header("Mixture of Agents", anchor=False)
st.write("A demo of the Mixture of Agents architecture proposed by Together AI, Powered by Groq LLMs.")
st.image("./static/moa_groq.svg", caption="Mixture of Agents Workflow", width=1000)

# Display current configuration
with st.expander("Current MOA Configuration", expanded=False):
    st.markdown(f"**Main Model**: ``{st.session_state.main_model}``")
    st.markdown(f"**Main Model Temperature**: ``{st.session_state.main_temp:.1f}``")
    st.markdown(f"**Layers**: ``{st.session_state.cycles}``")
    st.markdown(f"**Layer Agents Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.layer_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )

# Initialize session state for messages if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a form for the chat input
with st.form(key='chat_form'):
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            label="",
            placeholder="Ask a question",
            key="chat_input"
        )
    with col2:
        submit_button = st.form_submit_button("Submit")

    # Move these buttons inside the form
    col1, col2 = st.columns([2, 2])
    with col1:
        reprompt_button = st.form_submit_button("Re-Engineer Prompt")
    with col2:
        generate_button = st.form_submit_button("Generate Agents With AutoGroq™")

# Handle form submission
if submit_button or reprompt_button or generate_button:
    # Update the query in session state
    st.session_state.query = query

# Handle chat input submission
if submit_button and st.session_state.query:
    st.session_state.messages.append({"role": "user", "content": st.session_state.query})
    with st.chat_message("user"):
        st.write(st.session_state.query)

    moa_agent: MOAgent = st.session_state.moa_agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        ast_mess = stream_response(moa_agent.chat(st.session_state.query, output_format='json'))
        response = st.write_stream(ast_mess)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Handle prompt re-engineering
if reprompt_button:
    if not st.session_state.query:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Re-engineering prompt..."):
            rephrased_query = rephrase_prompt(st.session_state.query)
        st.session_state.messages.append({"role": "system", "content": f"Rephrased prompt: {rephrased_query}"})
        with st.chat_message("system"):
            st.write(f"Rephrased prompt: {rephrased_query}")
        
        # Use the rephrased query for the MOA agent
        moa_agent: MOAgent = st.session_state.moa_agent
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            ast_mess = stream_response(moa_agent.chat(rephrased_query, output_format='json'))
            response = st.write_stream(ast_mess)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if generate_button:
    st.write("AutoGroq agent generation initiated!")
    # Add your AutoGroq agent generation logic here