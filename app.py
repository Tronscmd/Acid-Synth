import streamlit as st
import numpy as np
from scipy.io import wavfile
import io

# --- FUNZIONI DSP (Il cuore del tuo synth) ---
def apply_lowpass_filter(data, cutoff_ratio):
    out = np.zeros_like(data)
    alpha = max(0.01, min(0.4, cutoff_ratio))
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

def generate_tone_adsr(freq, duration, velocity, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), False)
    wave = 0.5 * np.sign(np.sin(2 * np.pi * freq * t)) + 0.4 * (2 * (t * freq % 1) - 1)
    
    n_total = len(t)
    n_a, n_d, n_r = int(n_total * 0.05), int(n_total * 0.2), int(n_total * 0.1)
    n_s = n_total - n_a - n_d - n_r
    env = np.concatenate([np.linspace(0, 1, n_a), np.linspace(1, 0.3, n_d), 
                          np.full(n_s, 0.3), np.linspace(0.3, 0, n_r)])
    return wave * env * velocity

# --- INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="Python Acid Synth", page_icon="🎹")
st.title("🎹 Python Acid Synth")
st.markdown("Genera loop acidi direttamente nel browser e scaricali in WAV.")

# Sidebar per i preset e controlli globali
st.sidebar.header("🎛️ Controlli")
preset_name = st.sidebar.selectbox("Carica Preset", ["Custom", "Basso Techno", "Melodia Acid"])

# Logica Preset
if preset_name == "Basso Techno":
    d_root, d_bpm, d_pattern, d_cmax = 41.20, 130, "0, 0, 0, 0, 12, 0, 0, 0, 3, 0, 0, 0, 5, 0, 3, 2", 0.15
elif preset_name == "Melodia Acid":
    d_root, d_bpm, d_pattern, d_cmax = 110.0, 140, "0, 12, 0, 24, 12, 0, 7, 10, 12, 24, 12, 7, 3, 5, 0, 12", 0.35
else:
    d_root, d_bpm, d_pattern, d_cmax = 90.0, 120, "0, 12, 3, 7", 0.20

bpm = st.sidebar.slider("BPM", 60, 200, d_bpm)
root_note = st.sidebar.number_input("Frequenza Base (Hz)", value=d_root)
c_max = st.sidebar.slider("Cutoff Massimo Filtro", 0.01, 0.50, d_cmax)

pattern_input = st.text_area("Pattern (semitoni separati da virgola)", d_pattern)

if st.button("🚀 GENERA AUDIO"):
    try:
        pattern = [int(x.strip()) for x in pattern_input.split(",")]
        # Per semplicità usiamo velocity fissa a 1.0 nella web app o espandibile
        note_duration = (60 / bpm) / 4
        sample_rate = 44100
        
        full_sequence = []
        for i, step in enumerate(pattern):
            freq = root_note * (2 ** (step / 12))
            nota = generate_tone_adsr(freq, note_duration, 1.0)
            cutoff = 0.05 + (c_max - 0.05) * abs(np.sin(i * 0.4))
            full_sequence.append(apply_lowpass_filter(nota, cutoff))

        audio_buffer = np.concatenate(full_sequence)
        audio_buffer /= np.max(np.abs(audio_buffer))
        
        # Converti per WAV
        audio_int16 = (audio_buffer * 32767).astype(np.int16)
        
        # Buffer di memoria
        virtual_file = io.BytesIO()
        wavfile.write(virtual_file, sample_rate, audio_int16)
        
        st.audio(virtual_file)
        st.download_button(label="💾 Scarica .WAV", data=virtual_file, file_name="acid_loop.wav", mime="audio/wav")
        st.success("Audio generato! Clicca Play sopra.")
        
    except Exception as e:
        st.error(f"Errore nella generazione: {e}")