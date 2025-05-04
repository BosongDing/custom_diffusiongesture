# Personalize co-speech Gesture Generation

## Datasets

### Data Format and Model Requirements
The deep learning model requires following format of processed data for training and inference:

1. **Direction Vectors**: The model uses direction vectors representing bone orientations . These vectors capture the relative positioning of joints and are more effective for generating natural gestures. Each frame of motion is represented as a set of direction vectors between connected joints.

2. **Audio Features**: Two types of audio representations are used:
   - **Raw waveform**: The time-domain audio signal used for certain audio processing tasks.
   - **Mel-spectrograms**: Frequency-domain representations that capture the audio characteristics in a format suitable for neural networks.

3. **Sequence Data**: The model processes motion sequences of fixed length (typically 34 frames at 15 FPS), allowing it to learn the temporal dynamics of gestures.

4. **LMDB Database Structure**: Data is organized in an optimized Lightning Memory-Mapped Database (LMDB) format which stores:
   - Motion data as direction vectors
   - Aligned audio features
   - Auxiliary information like frame indices and timing
   
This LMDB structure allows efficient random access, batch loading, and helps manage the large dataset size during training.

The data processing pipeline transforms raw motion capture (BVH files) and audio into these required formats through several processing steps described for each dataset below.

### TED-Expressive
The TED-Expressive dataset contains 3D human motion sequences extracted from TED Talk videos. 

**Data Processing Pipeline:**
1. Raw motion data in BVH format is parsed to extract joint positions and rotations.
2. Joint positions are converted to direction vectors representing bone orientations.
3. These direction vectors are stored in NPZ files for each sequence.
4. Audio is extracted and processed to obtain raw waveforms and mel-spectrograms.
5. The NPZ files and audio features are combined and stored in an LMDB database for efficient access.

### Trinity
The Trinity dataset contains motion capture recordings of human gestures.

**Data Processing Pipeline:**
1. Raw BVH files contain motion capture data at 60 FPS.
2. The BVH files are parsed and converted to direction vectors using `BVH_convert.py`.
3. The direction vectors are stored in NPZ files with the suffix `_direction_vectors.npz`.
4. The data is resampled to 15 FPS during preprocessing.
5. Audio files (WAV) are processed to extract mel-spectrograms.
6. The `TrinityDataPreprocessor` combines motion and audio data and stores them in an LMDB database.
7. The LMDB database is accessed during training via the `TrinityLMDBDataset` class.

### BEAT
The BEAT dataset contains multi-modal motion capture data with aligned speech audio.

**Data Processing Pipeline:**
1. Raw BVH files are processed to extract skeletal motion.
2. The `convert_bvh_to_direction_vectors` function converts joint positions to direction vectors.
3. The direction vectors are stored in NPZ files with the naming pattern `filename_direction_vectors.npz`.
4. Audio is processed to extract spectrograms.
5. The preprocessor creates an LMDB database that combines motion and audio data.
6. During training, the dataset is loaded using a data loader that accesses the LMDB database.

## Usage Instructions
0. install the requirements 

 note that I am using CUDA 12.0, if you are using older versions you can follow the requirements in https://github.com/Advocate99/DiffGesture
```bash
conda create -n diffgesture python=3.7
conda activate diffgesture
pip install -r requirements.txt
```
1. first you can verify the BVH files contain correct data by visualizing it directly:
```bash
python bvh_player.py --bvh <path_to_your_bvh_file> --audio <path_to_your_audio_file>
```
2. then we need to convert BVH files into correct direction vector format. here we have 3 options for --dataset trinity/beat/all
```bash
python /scripts/preprocess/BVH_convert.py --dataset all
```
3. (optional)next, we convert the data collected by Quest3 into the same format.

3.1. (optional) visualize the collected data by running 
```bash
python /scripts/preprocess/play_json.py --json <path_to_your_json_file> --audio <path_to_your_audio_file>
```

3.2 for this step, you need to provide your huggingface token to download the pyannote model.\
you can get access to the model from https://huggingface.co/pyannote/voice-activity-detection \
after you sign in, you can get the token from https://huggingface.co/settings/tokens
```bash
echo 'export HUGGINGFACE_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```
preprocess the json and audio file into activate clips through pyannote
```bash
python /scripts/preprocess/preprocess.py
```

3.3 convert json and wav files into dataset
```bash
python /scripts/preprocess/json_convert.py
```
4.buid datasets

