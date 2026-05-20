import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wave
import io
import streamlit as st


st.title("Analyse FFT d'un signal audio")

audio_value = st.audio_input(
    "Enregistrer un son",
    sample_rate=44100
)

if audio_value is not None:
    audio_bytes = audio_value.getvalue()

    st.audio(audio_bytes, format="audio/wav")

    # Lecture du WAV en mémoire
    wav_buffer = io.BytesIO(audio_bytes)

    with wave.open(wav_buffer, "rb") as wf:
        fs = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw_audio = wf.readframes(n_frames)

    # Conversion bytes -> numpy
    if sample_width == 2:
        recording = np.frombuffer(raw_audio, dtype=np.int16)
        recording = recording.astype(np.float32) / 32768.0
    elif sample_width == 4:
        recording = np.frombuffer(raw_audio, dtype=np.int32)
        recording = recording.astype(np.float32) / 2147483648.0
    else:
        st.error("Format audio non supporté par ce script.")
        st.stop()

    # Si stéréo, on convertit en mono
    if n_channels > 1:
        recording = recording.reshape(-1, n_channels)
        recording = recording.mean(axis=1)

    # Suppression de la composante continue
    recording = recording - np.mean(recording)

    # Fenêtre pour réduire les fuites spectrales
    window = np.hanning(len(recording))
    recording_windowed = recording * window

    # FFT réelle, plus adaptée pour un signal réel
    fft = np.fft.rfft(recording_windowed)
    frequencies = np.fft.rfftfreq(len(recording_windowed), 1 / fs)

    magnitude = np.abs(fft)

    # On ignore la fréquence 0 Hz pour chercher la fondamentale
    if len(magnitude) > 1:
        idx_max = np.argmax(magnitude[1:]) + 1
        fondamental_freq = frequencies[idx_max]
    else:
        fondamental_freq = np.nan

    st.subheader("Résultat")
    st.write(f"Fréquence fondamentale estimée : **{fondamental_freq:.2f} Hz**")

    # Affichage temporel
    st.subheader("Signal temporel")

    t = np.arange(len(recording)) / fs

    fig_time, ax_time = plt.subplots()
    ax_time.plot(t, recording)
    ax_time.set_xlabel("Temps (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Signal enregistré")
    ax_time.grid(True)

    st.pyplot(fig_time)

    # Affichage FFT
    st.subheader("Spectre fréquentiel")

    fig_fft, ax_fft = plt.subplots()
    ax_fft.plot(frequencies, magnitude)
    ax_fft.set_title("Frequency Spectrum of Recorded Signal")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.set_xlim(0, fs / 2)
    ax_fft.grid(True)

    st.pyplot(fig_fft)