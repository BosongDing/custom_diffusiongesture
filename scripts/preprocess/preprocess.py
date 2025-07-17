from pyannote.audio import Pipeline
import librosa
import librosa.display
import soundfile as sf
import json
import os
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=huggingface_token)
file_path = "/home/bsd/cospeech/DiffGesture/data/quest/dataset/voice_20250511_173952.wav"
output = pipeline(file_path)
motion_data = json.load(open("/home/bsd/cospeech/DiffGesture/data/quest/dataset/AvatarMotionData_20250511_173952.json"))
motion_data = motion_data["frames"]
time_data = []
for frame in motion_data:
    time_data.append(frame["time"])
i = 0
for speech in output.get_timeline().support():
    print(speech.start, speech.end)
    if speech.end - speech.start < 2:
        continue
    #clip the audio to the speech segments
    y, sr = librosa.load(file_path, sr=None)
    y = y[int(speech.start * sr):int(speech.end * sr)]
    sf.write(file_path.replace(".wav", f"_speech_{i}.wav"), y, sr)
    
    #find the time of the speech in time_data
    start_index = 0
    end_index = 0
    for j in range(len(time_data)):
        
        if time_data[j] <= speech.start and time_data[j+1]>=speech.start:
            if abs(time_data[j] - speech.start) < abs(time_data[j+1] - speech.start):
                start_index = j
            else:
                start_index = j+1
            print(time_data[j], time_data[j+1])
        if time_data[j] <= speech.end and time_data[j+1]>=speech.end:
            if abs(time_data[j] - speech.end) < abs(time_data[j+1] - speech.end):
                end_index = j
            else:
                end_index = j+1
            print(time_data[j], time_data[j+1])
    print(start_index, end_index)
    
    #clip the motion data to the speech segments
    motion_data_segment = motion_data[start_index:end_index]
    #save the motion data in original json format
    output_json = {"frames": motion_data_segment}
    with open(file_path.replace(".wav", f"_motion_{i}.json"), "w") as f:
        json.dump(output_json, f, indent=4)
    i += 1

