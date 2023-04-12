from torch.utils.data import Dataset
import torch
import random
import os

ACTION_CLASSES = ['walk', 'sitDown', 'standUp', 'pickUp', 'carry', 
                  'throw', 'push', 'pull', 'waveHands', 'clapHands']
CLASS_SIZE = len(ACTION_CLASSES)

NUM_UNIQUE_JOINTS = 20

# `sub_seq_num` is same as `T` in the paper:
# "In our experiments, each video sequence is divided to T sub-sequences with the 
# same length, and one frame is randomly selected from each sub-sequence."
def read_UTKinect(data_folder, sub_seq_num):

    print('Reading UTKinect data...')

    videos = []
    labels = []
    row = 0
    interpolate_count = 0

    action_label_path = os.path.join(data_folder, 'actionLabel.txt')
    action_label_file = open(action_label_path, 'r')

    while True: 

        line = action_label_file.readline().strip()
        if not line:
            break

        # (CLASS_SIZE+1) lines per video. The first line is the file name, open it.
        # Read the content into a list first.
        joints_data = []
        if row % (CLASS_SIZE + 1) == 0: 
            joints_data_path = os.path.join(data_folder, 'joints', 'joints_' + line + '.txt')
            joints_data_file = open(joints_data_path, 'r')
            
            while True:
                data_entry = joints_data_file.readline().strip()
                if not data_entry:
                    break
                data_entry = data_entry.split("  ")
                timestamp = int(data_entry[0])
                # NUM_UNIQUE_JOINTS joints per timestamp, create a tensor for each joint
                # A frame is a list of NUM_UNIQUE_JOINTS tensors
                frame = [torch.Tensor([float(data_entry[3*j+1]), 
                                       float(data_entry[3*j+2]), 
                                       float(data_entry[3*j+3])]) for j in range(NUM_UNIQUE_JOINTS)]
                frame = [timestamp] + frame
                joints_data.append(frame)

            joints_data_file.close()
            row += 1

        # Each of the remaining CLASS_SIZE lines in action_label_file represents an action
        for _ in range(CLASS_SIZE):

            line = action_label_file.readline()
            # some values in the raw file are wrongly delimited with double space
            line = line.strip().replace(":", "").replace("  ", " ").split(" ")
            
            row += 1
            if line[1] == 'NaN': # some timestamps are labelled as NaN
                continue 

            label, startTime, endTime = ACTION_CLASSES.index(line[0]), int(line[1]), int(line[2])
            video = [[] for _ in range(sub_seq_num)] 

            for frame in joints_data:
                timestamp = frame[0]
                if startTime <= timestamp <= endTime:
                    # we want to map timestamp to an integer within [0, sub_seq_num-1]
                    t = int((timestamp - startTime) / (endTime - startTime) * sub_seq_num)
                    t = min(t, sub_seq_num - 1)
                    # no need the first item (timestamp)
                    video[t].append(frame[1:]) 
                if timestamp > endTime:
                    break # early exit, just to save some time
            
            # Interpolate if necessary. Sometimes a sub sequence can be empty, if the original
            # video is too short, or you chop the video into too many parts. If there is any 
            # empty bin, we take two frames adjacent to the bin and interpolate.
            for i, sub_seq in enumerate(video):
                if len(sub_seq) == 0:
                    
                    interpolate_count += 1  # for statistics calculation to be displayed

                    left_bin = i
                    while len(video[left_bin]) == 0 and left_bin != 0:
                        left_bin -= 1

                    right_bin = i
                    while len(video[right_bin]) == 0 and right_bin != sub_seq_num - 1:
                        right_bin += 1

                    # special handling if an non-empty bin can't be found on left or right
                    if left_bin == 0 and len(video[left_bin]) == 0:
                        right_frame = video[right_bin][0]
                        sub_seq.append(right_frame[:])
                    elif right_bin == sub_seq_num - 1 and len(video[right_bin]) == 0:
                        left_frame = video[left_bin][-1]
                        sub_seq.append(left_frame[:])
                    else: # take interpolation
                        new_frame = []
                        left_frame = video[left_bin][-1]
                        right_frame = video[right_bin][0]
                        for left_joint_pos, right_joint_pos in zip(left_frame, right_frame):
                            weighted_avg = (left_joint_pos * (right_bin - i) + right_joint_pos * 
                                (i - left_bin)) / (right_bin - left_bin)
                            new_frame.append(weighted_avg)
                        sub_seq.append(new_frame)

            videos.append(video)
            labels.append(label)
            
    # Display interpolation percentage. High percentage may mean you have chosen a sub_seq_num
    # that is too high.
    interpolate_percent = interpolate_count / (len(videos) * sub_seq_num)
    print(f'Interpolation percentage: {interpolate_percent:.2%}')

    return videos, labels


class UTKinectDataset(Dataset):

    def __init__(self, videos, labels, indexes, joints_order):
        self.videos = [videos[i] for i in indexes]
        self.labels = [torch.tensor(labels[i]) for i in indexes]
        self.joints_order = joints_order
        self.len = len(self.labels)

    def __getitem__(self, index):
        
        orig_video = self.videos[index]
        sampled_video = []
        for sub_seq in orig_video:
            # sample one frame from each sub-sequence
            rand = random.randint(0, len(sub_seq) - 1)
            frame = sub_seq[rand]
            # stack 3D locations of joints according to joints_order
            frame = torch.stack([frame[j-1] for j in self.joints_order])
            sampled_video.append(frame)

        return torch.stack(sampled_video), self.labels[index]
    
    def __len__(self):
        return self.len

