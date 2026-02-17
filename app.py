import streamlit as st
import numpy as np
from scipy.io import wavfile
import io

# --- FUNZIONI DSP ---
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

def detect_pitch_to_pattern(audio_data, sr, root_freq, steps=16):
    if np.max(np.abs(audio_data)) == 0: return "0"
    samples_per_step = len(audio_data) // steps
    detected_steps = []
    for i in range(steps):
        segment = audio_data[i*samples_per_step : (i+1)*samples_per_step]
        if np.max(np.abs(segment)) < 0.02:
            detected_steps.append(0)
            continue
        window = segment * np.hanning(len(segment))
        fft = np.fft.rfft(window)
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        idx = np.argmax(np.abs(fft))
        freq = freqs[idx]
        if freq > 0:
            semitones = round(12 * np.log2(freq / root_freq))
            detected_steps.append(int(np.clip(semitones, -24, 24)))
        else:
            detected_steps.append(0)
    return ", ".join(map(str, detected_steps))

# --- INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="Python Acid Synth", page_icon="🎹")
st.title("🎹 Python Acid Synth & Analyzer")

# 1. INIZIALIZZAZIONE SESSION STATE (Cruciale per il salvataggio)
if 'pattern_key' not in st.session_state:
    st.session_state.pattern_key = "0, 12, 3, 7"

# Sidebar
st.sidebar.header("🎛️ Controlli Synth")
preset_name = st.sidebar.selectbox("Carica Preset", ["Custom", "Basso Techno", "Melodia Acid"])

# Logica Preset rapida
if preset_name == "Basso Techno":
    d_root, d_bpm, d_cmax = 41.20, 130, 0.15
    if st.sidebar.button("Applica Preset Basso"):
        st.session_state.pattern_key = "0, 0, 0, 0, 12, 0, 0, 0, 3, 0, 0, 0, 5, 0, 3, 2"
elif preset_name == "Melodia Acid":
    d_root, d_bpm, d_cmax = 110.0, 140, 0.35
    if st.sidebar.button("Applica Preset Acid"):
        st.session_state.pattern_key = "0, 12, 0, 24, 12, 0, 7, 10, 12, 24, 12, 7, 3, 5, 0, 12"
else:
    d_root, d_bpm, d_cmax = 90.0, 120, 0.20

bpm = st.sidebar.slider("BPM", 60, 200, d_bpm)
root_note = st.sidebar.number_input("Frequenza Base (Hz)", value=d_root)
c_max = st.sidebar.slider("Cutoff Massimo Filtro", 0.01, 0.50, d_cmax)

# --- ANALIZZATORE AUDIO ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Analyzer")
uploaded_file = st.sidebar.file_uploader("Carica .wav", type=["wav", "WAV"])

if uploaded_file is not None:
    if st.sidebar.button("🔍 ESTRAI NOTE"):
        try:
            sr_up, data_up = wavfile.read(uploaded_file)
            if len(data_up.shape) > 1: data_up = data_up[:, 0]
            # Estrazione e salvataggio nello stato
            st.session_state.pattern_key = detect_pitch_to_pattern(data_up, sr_up, root_note)
            st.sidebar.success("Note caricate nel sequencer!")
        except Exception as e:
            st.sidebar.error(f"Errore: {e}")

# --- LAYOUT PRINCIPALE ---
col1, col2 = st.columns(2)

with col1:
    # Colleghiamo il widget direttamente alla chiave dello stato
    pattern_input = st.text_area("Pattern Sequencer (Modificabile)", 
                                 key="pattern_key", 
                                 help="Cambia i numeri per modificare la melodia")

    if st.button("🚀 GENERA AUDIO SYNTH"):
        try:
            # Usiamo st.session_state.pattern_key invece di pattern_input per sicurezza
            pattern = [int(x.strip()) for x in st.session_state.pattern_key.split(",")]
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
            
            audio_int16 = (audio_buffer * 32767).astype(np.int16)
            virtual_file = io.BytesIO()
            wavfile.write(virtual_file, sample_rate, audio_int16)
            
            st.audio(virtual_file)
            st.download_button(label="💾 Scarica .WAV", data=virtual_file, file_name="acid_synth.wav", mime="audio/wav")
        except Exception as e:
            st.error(f"Errore: {e}")

with col2:
    if uploaded_file is not None:
        st.write("🎹 **Sampler pronto**")
        if st.button("🎛️ FILTRA CAMPIONE"):
            # (Logica filtro come prima)
            sr_up, data_up = wavfile.read(uploaded_file)
            if len(data_up.shape) > 1: data_up = data_up[:, 0]
            audio_f = apply_lowpass_filter(data_up.astype(np.float32)/np.max(np.abs(data_up)), c_max)
            v_file = io.BytesIO()
            wavfile.write(v_file, sr_up, (audio_f * 32767).astype(np.int16))
            st.audio(v_file)

st.markdown("<br><hr><center>Copyright © 2026 VMMGAG</center>", unsafe_allow_html=True)