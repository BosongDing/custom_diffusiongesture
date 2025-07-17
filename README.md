# Personalize co-speech Gesture Generation

## Usage Instructions
0. install the requirements 

 note that I am using CUDA 12.0, if you are using older versions you can follow the requirements in https://github.com/Advocate99/DiffGesture
```bash
conda create -n diffgesture python=3.7
conda activate diffgesture
pip install -r requirements.txt
cd DiffGesture
export PYTHONPATH=$PYTHONPATH:$(pwd)
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

this python script will build the dataset and save it in the data folder
the script assumes the following structure:
```bash
DiffGesture/data/
├── beat_english_v0.2.1/
    ├──1
    ├──2
    ...
    ├──30
├── trinity
    ├── allRec
    ├── allRecAudio
    ├── allTestMotion
    ├── allTestAudio
├── quest
    ├── data1.npz
    ├── data1.wav
    ├── data2.npz
    ├── data2.wav
    ├── ...

```
the --dataset can be trinity_all/beat_all/beat_separated/all/quest
make sure you have enough space to store the dataset it will take around 1.2Tb for all the dataset(including the Ted-expressive dataset)
beat_all will take around 1.5h to build
```bash
python /scripts/preprocess/build_dataset.py --dataset all
```