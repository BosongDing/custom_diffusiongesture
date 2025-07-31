""" create data samples """
import logging
from collections import defaultdict

import lmdb
import math
import numpy as np
import pyarrow
import tqdm
from sklearn.preprocessing import normalize

import utils.data_utils_expressive
from data_loader.motion_preprocessor_expressive import MotionPreprocessor
def custom_serialize(data):
    """Convert numpy arrays to lists before serialization"""
    if isinstance(data, np.ndarray):
        try:
            return data.tolist()
        except Exception as e:
            print(f"Error converting numpy array to list: {e}")
            # Fallback: convert to smaller chunks
            if data.size > 1000000:  # If array is very large
                print(f"Large array detected with shape {data.shape}, converting in chunks")
                result = []
                chunk_size = 10000
                for i in range(0, data.shape[0], chunk_size):
                    end = min(i + chunk_size, data.shape[0])
                    result.extend(data[i:end].tolist())
                return result
            return data.tolist()  # Try again, may fail for the same reason
    elif isinstance(data, list):
        return [custom_serialize(item) for item in data]
    elif isinstance(data, dict):
        return {k: custom_serialize(v) for k, v in data.items()}
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        try:
            # For other types, try to convert to a basic Python type
            return data.__dict__  # For custom objects with __dict__
        except:
            try:
                return str(data)  # Last resort: convert to string
            except:
                print(f"Could not serialize object of type {type(data)}")
                return None  # If all else fails

class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride,
                 pose_resampling_fps, mean_pose, mean_dir_vec, disable_filtering=False):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_pose = mean_pose
        self.mean_dir_vec = mean_dir_vec
        self.disable_filtering = disable_filtering

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.spectrogram_sample_length = utils.data_utils_expressive.calc_spectrogram_length_from_motion_length(self.n_poses, self.skeleton_resampling_fps)
        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 200  # in MB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

    def run(self):
        n_filtered_out = defaultdict(int)
        
        # Use a read-only transaction that won't conflict with other transactions
        try:
            src_txn = self.src_lmdb_env.begin(write=False)
            
            # sampling and normalization
            cursor = src_txn.cursor()
            for key, value in cursor:
                video = pyarrow.deserialize(value)
                vid = video['vid']
                clips = video['clips']
                for clip_idx, clip in enumerate(clips):
                    try:
                        filtered_result = self._sample_from_clip(vid, clip)
                        for type in filtered_result.keys():
                            n_filtered_out[type] += filtered_result[type]
                    except Exception as e:
                        print(f"Error processing clip {vid}, {clip_idx}: {e}")
                        break
                        
        except Exception as e:
            print(f"Error in run method: {e}")
        finally:
            # Explicitly close the cursor if it exists
            if 'cursor' in locals() and cursor is not None:
                cursor.close()
            
            # Ensure the transaction is closed properly
            if 'src_txn' in locals() and src_txn is not None:
                src_txn.abort()  # Since it's read-only, we can just abort it

        # print stats
        try:
            with self.dst_lmdb_env.begin() as txn:
                print('no. of samples: ', txn.stat()['entries'])
                n_total_filtered = 0
                for type, n_filtered in n_filtered_out.items():
                    # print('{}: {}'.format(type, n_filtered))
                    n_total_filtered += n_filtered
                print('no. of excluded samples: {} ({:.1f}%)'.format(
                    n_total_filtered, 100 * n_total_filtered / (txn.stat()['entries'] + n_total_filtered)))
        except Exception as e:
            print(f"Error getting stats: {e}")

        # close db
        try:
            self.src_lmdb_env.close()
            self.dst_lmdb_env.sync()
            self.dst_lmdb_env.close()
        except Exception as e:
            print(f"Error closing databases: {e}")

    def _sample_from_clip(self, vid, clip):
        clip_skeleton = clip['skeletons_3d']
        clip_audio = clip['audio_feat']
        clip_audio_raw = clip['audio_raw']
        clip_word_list = clip['words']
        clip_s_f, clip_e_f = clip['start_frame_no'], clip['end_frame_no']
        clip_s_t, clip_e_t = clip['start_time'], clip['end_time']

        n_filtered_out = defaultdict(int)

        # skeleton resampling
        clip_skeleton = utils.data_utils_expressive.resample_pose_seq(clip_skeleton, clip_e_t - clip_s_t, self.skeleton_resampling_fps)

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_words_list = []
        sample_audio_list = []
        sample_spectrogram_list = []

        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1
        expected_audio_length = utils.data_utils_expressive.calc_spectrogram_length_from_motion_length(len(clip_skeleton), self.skeleton_resampling_fps)
        assert abs(expected_audio_length - clip_audio.shape[1]) <= 5, 'audio and skeleton lengths are different'

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            subdivision_start_time = clip_s_t + start_idx / self.skeleton_resampling_fps
            subdivision_end_time = clip_s_t + fin_idx / self.skeleton_resampling_fps
            sample_words = self.get_words_in_time_range(word_list=clip_word_list,
                                                        start_time=subdivision_start_time,
                                                        end_time=subdivision_end_time)

            # spectrogram
            audio_start = math.floor(start_idx / len(clip_skeleton) * clip_audio.shape[1])
            audio_end = audio_start + self.spectrogram_sample_length
            if audio_end > clip_audio.shape[1]:  # correct size mismatch between poses and audio
                # logging.info('expanding audio array, audio start={}, end={}, clip_length={}'.format(
                #     audio_start, audio_end, clip_audio.shape[1]))
                n_padding = audio_end - clip_audio.shape[1]
                padded_data = np.pad(clip_audio, ((0, 0), (0, n_padding)), mode='symmetric')
                sample_spectrogram = padded_data[:, audio_start:audio_end]
            else:
                sample_spectrogram = clip_audio[:, audio_start:audio_end]

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            if audio_end > len(clip_audio_raw):  # correct size mismatch between poses and audio
                # logging.info('expanding audio array, audio start={}, end={}, clip_length={}'.format(
                #     audio_start, audio_end, len(clip_audio_raw)))
                n_padding = audio_end - len(clip_audio_raw)
                padded_data = np.pad(clip_audio_raw, (0, n_padding), mode='symmetric')
                sample_audio = padded_data[audio_start:audio_end]
            else:
                sample_audio = clip_audio_raw[audio_start:audio_end]

            if len(sample_words) >= 2:
                # filtering motion skeleton data
                sample_skeletons, filtering_message = MotionPreprocessor(sample_skeletons, self.mean_pose).get()
                is_correct_motion = (sample_skeletons != [])
                motion_info = {'vid': vid,
                               'start_frame_no': clip_s_f + start_idx,
                               'end_frame_no': clip_s_f + fin_idx,
                               'start_time': subdivision_start_time,
                               'end_time': subdivision_end_time,
                               'is_correct_motion': is_correct_motion, 'filtering_message': filtering_message}

                if is_correct_motion or self.disable_filtering:
                    sample_skeletons_list.append(sample_skeletons)
                    sample_words_list.append(sample_words)
                    sample_audio_list.append(sample_audio)
                    sample_spectrogram_list.append(sample_spectrogram)
                    aux_info.append(motion_info)
                else:
                    n_filtered_out[filtering_message] += 1

        if len(sample_skeletons_list) > 0:
            # Create a single transaction outside the loop
            txn = None
            try:
                txn = self.dst_lmdb_env.begin(write=True)
                for words, poses, audio, spectrogram, aux in zip(sample_words_list, sample_skeletons_list,
                                                                 sample_audio_list, sample_spectrogram_list,
                                                                 aux_info):
                    # preprocessing for poses
                    poses = np.asarray(poses)
                    dir_vec = utils.data_utils_expressive.convert_pose_seq_to_dir_vec(poses)
                    normalized_dir_vec = self.normalize_dir_vec(dir_vec, self.mean_dir_vec)

                    # Ensure all numpy arrays have correct dtypes
                    poses = poses.astype(np.float32)
                    normalized_dir_vec = normalized_dir_vec.astype(np.float32)
                    audio = np.asarray(audio, dtype=np.float32)
                    spectrogram = np.asarray(spectrogram, dtype=np.float32)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [words, poses, normalized_dir_vec, audio, spectrogram, aux]
                    try:
                        serializable_v = [words, poses.tolist(), normalized_dir_vec.tolist(), audio.tolist(), spectrogram.tolist(), aux]
                        v = pyarrow.serialize(serializable_v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
                    except Exception as e:
                        print(f"Serialization error: {e}")
                        # print(f"Types: {[type(x) for x in v]}")
                        # print(f"Shapes: poses {poses.shape}, dir_vec {normalized_dir_vec.shape}, audio {audio.shape}, spec {spectrogram.shape}")
                
                # Commit the transaction only once after all items are processed
                txn.commit()
                txn = None
            except Exception as e:
                print(f"Transaction error: {e}")
                if txn is not None:
                    txn.abort()  # Make sure to abort on error
            finally:
                if txn is not None:
                    txn.abort()  # Safety check in case transaction wasn't committed or aborted
                    
        return n_filtered_out

    @staticmethod
    def normalize_dir_vec(dir_vec, mean_dir_vec):
        return dir_vec - mean_dir_vec

    @staticmethod
    def get_words_in_time_range(word_list, start_time, end_time):
        words = []

        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]

            if word_s >= end_time:
                break

            if word_e <= start_time:
                continue

            words.append(word)

        return words

    @staticmethod
    def unnormalize_data(normalized_data, data_mean, data_std, dimensions_to_ignore):
        """
        this method is from https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
        """
        T = normalized_data.shape[0]
        D = data_mean.shape[0]

        origData = np.zeros((T, D), dtype=np.float32)
        dimensions_to_use = []
        for i in range(D):
            if i in dimensions_to_ignore:
                continue
            dimensions_to_use.append(i)
        dimensions_to_use = np.array(dimensions_to_use)

        origData[:, dimensions_to_use] = normalized_data

        # potentially inefficient, but only done once per experiment
        stdMat = data_std.reshape((1, D))
        stdMat = np.repeat(stdMat, T, axis=0)
        meanMat = data_mean.reshape((1, D))
        meanMat = np.repeat(meanMat, T, axis=0)
        origData = np.multiply(origData, stdMat) + meanMat

        return origData
