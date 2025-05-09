import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Inisialisasi model dan tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

# Simpan riwayat percakapan
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_user_inputs" not in st.session_state:
    st.session_state.past_user_inputs = []

st.title("Chatbot DialoGPT")
st.write("Chatbot ini menggunakan model DialoGPT dari Hugging Face. Ketik sesuatu!")

user_input = st.text_input("Anda:", key="input")

if user_input:
    # Tokenisasi
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Gabungkan dengan history
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate respons
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode respons
    bot_output = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Simpan dan tampilkan chat
    st.session_state.past_user_inputs.append((user_input, bot_output))

for user, bot in reversed(st.session_state.past_user_inputs):
    st.markdown(f"*Anda:* {user}")
    st.markdown(f"*Bot:* {bot}")