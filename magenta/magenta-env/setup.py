import os
import subprocess
import shutil
import zipfile
import sys
import site

def install_package(package):
    """Install a Python package using pip with --user flag"""
    print(f"Installing {package}...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', package], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        sys.exit(1)

def unzip_file(zip_path, extract_path):
    """Extract a zip file to the specified path"""
    print(f'Extracting {zip_path} to {extract_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def setup_fluidsynth_path():
    """Setup FluidSynth path in Python environment"""
    fluidsynth_path = r'C:\Users\USER\Desktop\Victor\fluidsynth-2.4.6-win10-x64\bin'
    
    # Create a .pth file in site-packages to add FluidSynth to Python's DLL search path
    site_packages = site.getsitepackages()[0]
    pth_file = os.path.join(site_packages, 'fluidsynth.pth')
    
    with open(pth_file, 'w') as f:
        f.write(fluidsynth_path)
    
    # Set environment variable for the current session
    os.environ['PATH'] = fluidsynth_path + os.pathsep + os.environ['PATH']
    print(f"Added FluidSynth path: {fluidsynth_path}")

print('Setting up FluidSynth...')
setup_fluidsynth_path()

print('Copying checkpoints from GCS...')
checkpoint_dir = 'onsets-frames'

# Remove and create directory
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
os.makedirs(checkpoint_dir)

# Copy checkpoints using gsutil
gsutil_command = f'gsutil -q -m cp -R gs://magentadata/models/onsets_frames_transcription/*checkpoint*.zip {checkpoint_dir}/'
subprocess.run(gsutil_command, shell=True, check=True)

# Unzip checkpoints using Python's zipfile
maestro_zip = os.path.join(checkpoint_dir, 'maestro_checkpoint.zip')
maestro_dir = os.path.join(checkpoint_dir, 'maestro')
if os.path.exists(maestro_zip):
    unzip_file(maestro_zip, maestro_dir)
MAESTRO_CHECKPOINT_DIR = os.path.join(maestro_dir, 'train')

egmd_zip = os.path.join(checkpoint_dir, 'e-gmd_checkpoint.zip')
egmd_dir = os.path.join(checkpoint_dir, 'e-gmd')
if os.path.exists(egmd_zip):
    unzip_file(egmd_zip, egmd_dir)
EGMD_CHECKPOINT_DIR = egmd_dir

print('Installing dependencies...')

# Python dependencies - install one by one with --user flag
# Modify your packages list to include specific versions
packages = [
    'numpy<2',
    'pyfluidsynth==1.3.2',  # Specific version known to work
    'pretty_midi',
    'magenta[tensorflow]'
]

for package in packages:
    install_package(package)

print('Setup complete.')

#@title Select Model
model_type = "E-GMD (Drums)" #@param ["MAESTRO (Piano)", "E-GMD (Drums)"]


#@title Initialize Model
import tensorflow.compat.v1 as tf
import librosa
import numpy as np

from magenta.common import tf_utils
from note_seq import audio_io
from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
import note_seq
from note_seq import midi_io
from note_seq import sequences_lib

tf.disable_v2_behavior()

## Define model and load checkpoint
## Only needs to be run once.

if model_type.startswith('MAESTRO'):
  config = configs.CONFIG_MAP['onsets_frames']
  hparams = config.hparams
  hparams.use_cudnn = False
  hparams.batch_size = 1
  checkpoint_dir = MAESTRO_CHECKPOINT_DIR
elif model_type.startswith('E-GMD'):
  config = configs.CONFIG_MAP['drums']
  hparams = config.hparams
  hparams.batch_size = 1
  checkpoint_dir = EGMD_CHECKPOINT_DIR
else:
  raise ValueError('Unknown Model Type')

examples = tf.placeholder(tf.string, [None])

dataset = data.provide_batch(
    examples=examples,
    preprocess_examples=True,
    params=hparams,
    is_training=False,
    shuffle_examples=False,
    skip_n_initial_records=0)

estimator = train_util.create_estimator(
    config.model_fn, checkpoint_dir, hparams)

iterator = tf.data.make_initializable_iterator(dataset)
next_record = iterator.get_next()



# Replace the file upload section with direct file loading
wav_path = os.path.abspath(os.path.join('..', '..', 'sounds', 'drums_file.wav'))
if not os.path.exists(wav_path):
    raise FileNotFoundError(f"Audio file not found: {wav_path}")
print(f'Loading audio file: {wav_path}')

with open(wav_path, 'rb') as f:
    wav_data = f.read()

to_process = []
example_list = list(
    audio_label_data_utils.process_record(
        wav_data=wav_data,
        sample_rate=hparams.sample_rate,
        ns=note_seq.NoteSequence(),
        example_id=os.path.basename(wav_path),
        min_length=0,
        max_length=-1,
        allow_empty_notesequence=True))
assert len(example_list) == 1
to_process.append(example_list[0].SerializeToString())

print('Processing complete for', wav_path)

sess = tf.Session()

sess.run([
    tf.initializers.global_variables(),
    tf.initializers.local_variables()
])

sess.run(iterator.initializer, {examples: to_process})

def transcription_data(params):
  del params
  return tf.data.Dataset.from_tensors(sess.run(next_record))
input_fn = infer_util.labels_to_features_wrapper(transcription_data)

#@title Run inference
prediction_list = list(
    estimator.predict(
        input_fn,
        yield_single_examples=False))
assert len(prediction_list) == 1

sequence_prediction = note_seq.NoteSequence.FromString(
    prediction_list[0]['sequence_predictions'][0])

# Ignore warnings caused by pyfluidsynth
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Save MIDI file
midi_filename = 'prediction.mid'
midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
print(f'Saved MIDI file as: {midi_filename}')

# Optional: Play the sequence (if FluidSynth is properly configured)
try:
    note_seq.play_sequence(
        sequence_prediction,
        note_seq.midi_synth.fluidsynth,
        colab_ephemeral=False
    )
except Exception as e:
    print(f"Could not play sequence: {e}")
    print("MIDI file was still saved successfully.")

# Visualize the sequence using matplotlib instead of bokeh
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
notes = [(note.start_time, note.pitch) for note in sequence_prediction.notes]
times, pitches = zip(*notes) if notes else ([], [])
plt.scatter(times, pitches, alpha=0.5)
plt.xlabel('Time (seconds)')
plt.ylabel('MIDI Pitch')
plt.title('Transcribed Notes')
plt.grid(True)
plt.show()