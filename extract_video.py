import pandas as pd
import os
import cv2
from moviepy.editor import VideoFileClip

#data = "/home/felix.wernlein/project/datasets/mosi/label.csv"
#dataset_path = "/home/felix.wernlein/project/datasets/mosi/Raw"
#feature_path = "/home/felix.wernlein/project/datasets/mosi"

"""data = "/home/felix.wernlein/project/datasets/mosei/label.csv"
dataset_path = "/home/felix.wernlein/project/datasets/mosei/Raw"
feature_path = "/home/felix.wernlein/project/datasets/mosei"""

dataset_path = "/home/felix.wernlein/project/datasets/meld/"
os.path.join(dataset_path, f"frames_train/frame_count.png")
# dia{Dialogue_ID}_utt{Utterance_ID}.mp4
df_train = pd.read_csv(dataset_path + "train_sent_emo.csv")
df_test = pd.read_csv(dataset_path + "test_sent_emo.csv")
df_dev = pd.read_csv(dataset_path + "dev_sent_emo.csv")

df_train['video_paths'] = df_train.apply(lambda x: f"{dataset_path}train/dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}.mp4", axis=1)
df_test['video_paths'] = df_test.apply(lambda x: f"{dataset_path}test/dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}.mp4", axis=1)
df_dev['video_paths'] = df_dev.apply(lambda x: f"{dataset_path}dev/dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}.mp4", axis=1)
def generate_frames(path, data_path, feature_type, frame_count, num_frames=15):
    images = []
    video_capture = cv2.VideoCapture(path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames // num_frames == 0:
        interval = 1
    else:
        # Calculate the interval between frames to sample
        interval = total_frames // num_frames

    current_frame = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if current_frame % interval == 0:
            frame_path = os.path.join(data_path, f'frames_{feature_type}/{frame_count:07d}.png')
            cv2.imwrite(frame_path, frame)
            images.append(frame_path)
            frame_count += 1
            if len(images) >= num_frames:
                break

        current_frame += 1

    video_capture.release()
    return images, frame_count


def generate_audio(path, data_path, feature, file_count):
    try:
        video = VideoFileClip(path)
        if video.audio is None:
            raise ValueError("No audio track found in video.")
        
        audio_path = os.path.join(data_path, f"audios_{feature}/audio_{file_count:05d}.wav")
        
        # Ensure that the video duration is within a valid range
        if video.duration is None or video.duration == 0:
            raise ValueError("Invalid video duration.")
        
        video.audio.write_audiofile(audio_path)
        file_count += 1
        return audio_path, file_count
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, file_count
def process_csv(data, feature_type):
    data['frames'] = [[] for _ in range(len(data))]
    data['audios'] = [[] for _ in range(len(data))]

    frame_counter, audio_counter = 0, 0

    video_paths = data['video_paths']
    for idx, video_path in enumerate(video_paths):
        frames, frame_counter = generate_frames(video_path, dataset_path, feature_type, frame_counter)
        audio, audio_counter = generate_audio(video_path, dataset_path, feature_type, audio_counter)

        data.at[idx, 'frames'] = ','.join(frames)
        if audio:
            data.at[idx, 'audio'] = audio

    data.to_csv(os.path.join(dataset_path, f'{feature_type}_sent_emo.csv'), index=False)
        

#process_csv(df_train, 'train')
process_csv(df_test, 'test')
process_csv(df_dev, 'dev')
