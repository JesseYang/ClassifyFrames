import os
train_video_path = 'raw_videos/data_20180206'

f = open('videolist.txt', 'w')
for (dirpath, dirnames, filenames) in os.walk(train_video_path):
    for filename in filenames:
        f.write(dirpath + '/' + filename+'\n')
