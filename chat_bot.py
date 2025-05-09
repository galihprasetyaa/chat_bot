import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Tentukan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model dan tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

# Inisialisasi session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_user_inputs" not in st.session_state:
    st.session_state.past_user_inputs = []

st.title("🧠 Chatbot DialoGPT")
st.write("Tanya apa saja ke chatbot ini!")

# Input pengguna
user_input = st.text_input("Anda:", key="input")

if user_input:
    # Tokenisasi dan pindah ke device
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

    # Gabungkan dengan riwayat chat
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate respons
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode jawaban bot
    bot_output = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Simpan chat
    st.session_state.past_user_inputs.append((user_input, bot_output))

# Tampilkan percakapan
for user, bot in reversed(st.session_state.past_user_inputs):
    st.markdown(f"**Anda:** {user}")
    st.markdown(f"**Bot:** {bot}")
