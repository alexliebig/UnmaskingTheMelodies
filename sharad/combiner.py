import numpy as np
import librosa
import soundfile as sf

filename1 = '../SampleWavFiles/vocals.mp3'
filename2 = '../SampleWavFiles/guitar.mp3'

# Load the audio files
y1, sr1 = librosa.load(filename1)
y2, sr2 = librosa.load(filename2)

y = np.vstack((y1, y2))

# Write the output to a file
sf.write('../SampleWavFiles/combined.wav', y.T, sr1)