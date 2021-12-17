#!/usr/bin/env python
#https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/simple_audio.ipynb#scrollTo=2-rayb7-3Y0I
#https://stackoverflow.com/questions/60032983/record-voice-with-recorder-js-and-upload-it-to-python-flask-server-but-wav-file
# -*- coding: utf-8 -*-
from flask import Flask
from flask import request
from flask import render_template
import os
import tensorflow as tf
import pickle
import pathlib

model = None
model_file = pathlib.Path("final.sav")
commands = ['go' 'up' 'right' 'left' 'yes' 'no' 'down' 'stop']
AUTOTUNE = tf.data.AUTOTUNE
def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary, desired_channels=1, desired_samples=176400)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def get_spectrogram_and_label_id(audio, label):
  global commands
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

def preprocess_dataset(files):
  global AUTOTUNE
  files_ds = tf.data.Dataset.from_tensor_slices(files)

  #files_d = tf.squeeze(files_ds, axis=[-1])
  print("1*********************1")
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  print("2*********************2")
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  print("3*********************3")
  return output_ds


app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    global model
    global model_file
    if request.method == "POST":
        f = request.files['audio_data']
        with open('01bb6a2a_nohash_0.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')
        sample_file = '01bb6a2a_nohash_0.wav'
        if model == None:
            model = pickle.load(open("final.sav", 'rb'))
        print("Model Loaded***************")
        sample_ds = preprocess_dataset([str(sample_file)])
        print(sample_ds)
        for spectrogram, label in sample_ds.batch(1):
            prediction = model(spectrogram)
            print(prediction)

        return render_template('index.html', request="POST")
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port='8080')
