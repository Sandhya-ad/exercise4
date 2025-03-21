# Repo for Exercise 4 cmput 412

First run `dts devel build -H csc22907.local -f` to build the executables on the duckie bot

## Part 1
To view just the april tag detection, run the command `dts devel run -H csc22907.local -L view-pre-pross-image`
Then in another terminal run command `dts start_gui_tool` then run `rqt_image_view` to view the image under the topic `/camera_apriltag/image/compressed`

For the behaviour based on april tag, run command:
`dts devel run -H csc22907.local -L april-tag-behavior`
The program will stop after execting the behaviour once. To see different behaviour run the program again with different tag.


## Part 2
Run the command
```
dts devel run -H csc22907.local -L crosswalk-behavior
```
The program will run twice to show 2 different behaviour: with the peDuckstrian and without peDuckstrian
To see just the peDuckstrian detection in the file `packages/my_packages/src/detect_blue_check.py` uncomment the line  
`#self.peduck_detector = PeDuckstrianDetector(True)  # 🚶‍♂️ Detects PeDuckstrians`
Then in another terminal run command `dts start_gui_tool` then run `rqt_image_view` to view the image under the topic `/camera_ducks/image/compressed`
And when there are ducks in front, you should see the bounding box around them

## Part 3
Run the command
```
dts devel run -H csc22907.local -L part3
```

To see just the duckiebot detection in the file `packages/my_packages/src/detect_blue_check.py` uncomment the line  
`#self.duckiebot_detector = DuckiebotDetector(True)`
Then in another terminal run command `dts start_gui_tool` then run `rqt_image_view` to view the image under the topic `camera_ducks/image/compressed`
This detects the black dots pattern in the back of the duckiebot so you should see the dots being contoured in the topic
