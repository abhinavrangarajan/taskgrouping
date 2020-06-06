import os
import sys
import random
import copy

N = int(sys.argv[2])

task_folder = sys.argv[1]
names = []
for building in os.listdir(task_folder):
    if os.path.isdir(task_folder + building):
        names.append([building, len(os.listdir(task_folder + building))])

names.sort(key=lambda x: x[1])
print("Buildings:", names)

# Create N dataset
tot = 0
num = 0
num_buildings = len(names)
train_dataset = set()
while tot < N and num < num_buildings:
    possible_buildings = []
    for i in range(num_buildings):    
        if names[i][0] not in train_dataset and tot + names[i][1] < N:
            possible_buildings.append(i)
    if len(possible_buildings) == 0:
        break
    new = random.choice(possible_buildings)
    train_dataset.add(names[new][0])
    tot += names[new][1]
    num += 1

tot0 = tot
print("Created N training dataset with {} images (from {} buildings).".format(tot, num))

# Create now 5N dataset
train5_dataset = copy.copy(train_dataset)

while tot < 5*N and num < num_buildings:
    possible_buildings = []
    for i in range(num_buildings):    
        if names[i][0] not in train5_dataset and tot + names[i][1] < 5*N:
            possible_buildings.append(i)
    if len(possible_buildings) == 0:
        break
    new = random.choice(possible_buildings)
    train5_dataset.add(names[new][0])
    tot += names[new][1]
    num += 1

print("Created 5N training dataset with {} images (from {} buildings).".format(tot, num))


# Create the val split
if num == num_buildings:
    print("ERROR rerun the script.")
    sys.exit(1)

possible_buildings = []
for i in range(num_buildings):
    if names[i][0] not in train_dataset and names[i][0] not in train5_dataset:
        possible_buildings.append(i)

val = random.choice(possible_buildings)
val_building = names[val][0]


# Write now these to file
with open("trainN_models.txt", "w") as f:
    f.write("\n".join(train_dataset))

with open("train5N_models.txt", "w") as f:
    f.write("\n".join(train5_dataset))

with open("val_models.txt", "w") as f:
    f.write(val_building)


print("\nN dataset has {} labels. 5N dataset has {} labels.".format(tot0, tot))
