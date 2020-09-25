# head-pose-estimation

Mini-Project during Summer 2020

Used a facial landmark detector and PnP algorithm to estimate orientation of head.
Takes live video feed and estimates Euler angles representing head orientation.

Motivation:
Many applications or games based on physical motion sensing, e.g. Kinect, require the user to be a sufficient distance away from the sensor and in a sufficiently large room.  In tighter spaces, such as at a desk setting, head pose estimation offers a much more feasible method of human-computer interaction.
The results of this project have be extended to allow users to control application or games in a convenient and intuitive manner.

Credits:
- inspired by PyImageSearch tutorial on facial landmark detection
- using dlib's pre-trained facial landmark detector
- using 3D facial model from https://github.com/yinguobing/head-pose-estimation/blob/master/assets/model.txt
