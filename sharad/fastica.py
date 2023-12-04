
# coding: utf-8

# In[78]:

import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
from sklearn.decomposition import FastICA
import numpy as np
import soundfile as sf


# In[79]:

filename = '../SampleWavFiles/mix.mp3'
audio, sr = librosa.load(filename)

print('Audio Shape:', audio.shape)
print('Sample Rate:', sr)


# In[80]:

ipd.Audio(audio, rate=sr)


# In[81]:

# Plot the audio signal
plt.figure(figsize=(14, 5))
plt.plot(audio)
plt.title('Audio Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


# In[82]:

# Plot the spectrogram
X = librosa.stft(audio)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('Spectrogram')
plt.show()


# In[96]:

# get magnitude and phase of the signal
mag, phase = librosa.magphase(X)

# Perform FastICA
n_components = 5
ica = FastICA(n_components=n_components, whiten="arbitrary-variance")
components = ica.fit_transform(mag)
print('Components Shape:', components.shape)


# In[98]:

# plot the components
plt.figure(figsize=(14, 5))
for i in range(n_components):
    plt.subplot(n_components, 1, i+1)
    plt.plot(components[:, i])
    plt.title('Component ' + str(i+1))
plt.tight_layout()
plt.show()


# In[111]:

# reconstruct and plot each component
plt.figure(figsize=(14, 5))
for i in range(n_components):
    # reconstruct the signal
    reconstructed = components[:, i:i+1] * phase
    reconstructed_signal = librosa.istft(reconstructed)
    plt.subplot(n_components, 1, i+1)
    plt.plot(reconstructed_signal)
    plt.title('Component ' + str(i+1))
    # save the reconstructed signal to a file
    sf.write('../SampleWavFiles/reconstructed' + str(i+1) + '.wav', reconstructed_signal, sr)
plt.tight_layout()
plt.show()

