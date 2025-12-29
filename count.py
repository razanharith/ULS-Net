import os

# List of directories to count images in
directories = [
    '/DATA/home/zyw/Razan/Datasets/ph2-new/test',
    '/DATA/home/zyw/Razan/Datasets/ph2-new/test_lab',
    '/DATA/home/zyw/Razan/Datasets/ph2-new/train',
    '/DATA/home/zyw/Razan/Datasets/ph2-new/train_lab',
    '/DATA/home/zyw/Razan/Datasets/ph2-new/valid',
    '/DATA/home/zyw/Razan/Datasets/ph2-new/valid_lab'
]

# Initialize a counter
total_count = 0

# Iterate through each directory and count images
for directory in directories:
    # Count all .jpg and .png files
    count = len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))])
    total_count += count
    print(f"Count in {directory}: {count}")

print(f"Total count of images across all directories: {total_count}")