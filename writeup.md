# Writeup: Mid-term Project - 3D Object Detection 
# 
This project is a midterm project in SDCN which is approached using a deep-learning approach is used to detect vehicles in LiDAR data based on a birds-eye view perspective of the 3D point-cloud. Also, a series of performance measures is used to evaluate the performance of the detection approach. The tasks in this part make up the mid-term project.

Detail information about file structure and installation instructions: [README.md](README.md)

This project consists of following tasks:
1. Compute Lidar Point-Cloud from Range Image
    1.1. Visualize range image channels (ID_S1_EX1)
        - Convert range image “range”, "intensity" channel to 8bit
        - Crop range image to +/- 90 deg. left and right of the forward-facing x-axis
        - Stack cropped range and intensity image vertically and visualize the result using OpenCV

    1.2. Visualize point-cloud (ID_S1_EX2)
        - Visualize the point-cloud using the open3d module
        - Find 6 examples of vehicles with varying degrees of visibility in the point-cloud
        - Try to identify vehicle features that appear stable in most of the inspected examples and describe them

2. Create Birds-Eye View from Lidar PCL
    2.1. Convert sensor coordinates to bev-map coordinates (ID_S2_EX1)
        - Convert coordinates in x,y [m] into x,y [pixel] based on width and height of the bev map

    2.2. Compute intensity layer of bev-map (ID_S2_EX2)
        - Assign lidar intensity values to the cells of the bird-eye view map
        - Adjust the intensity in such a way that objects of interest (e.g. vehicles) are clearly visible

    2.3. Compute height layer of bev-map (ID_S2_EX3)
        - Make use of the sorted and pruned point-cloud `lidar_pcl_top` from the previous task
        - Normalize the height in each BEV map pixel by the difference between max. and min. height
        - Fill the "height" channel of the BEV map with data from the point-cloud

3. Model-based Object Detection in BEV Image
    3.1. Add a second model from a GitHub repo (ID_S3_EX1)
        - In addition to Complex YOLO, extract the code for output decoding and post-processing from the GitHub repo.

    3.2. Extract 3D bounding boxes from model response (ID_S3_EX2)
        - Transform BEV coordinates in [pixels] into vehicle coordinates in [m]
        - Convert model output to expected bounding box format [class-id, x, y, z, h, w, l, yaw]

4. Performance Evaluation for Object Detection
    4.1 Compute intersection-over-union (IOU) between labels and detections (ID_S4_EX1)
        - For all pairings of ground-truth labels and detected objects, compute the degree of geometrical overlap
        - The function tools.compute_box_corners returns the four corners of a bounding box which can be used with the Polygon structure of the Shapely toolbox
        - Assign each detected object to a label only if the IOU exceeds a given threshold
        - In case of multiple matches, keep the object/label pair with max. IOU
        - Count all object/label-pairs and store them as “true positives”

    4.2 Compute false-negatives and false-positives (ID_S4_EX2)
        - Compute the number of false-negatives and false-positives based on the results from IOU and the number of ground-truth labels

    4.3 Compute precision and recall (ID_S4_EX3)
        - Compute “precision” over all evaluated frames using true-positives and false-positives
        - Compute “recall” over all evaluated frames using true-positives and false-negatives

## Step 1: Compute Lidar Point-Cloud from Range Image

### 1.1 Visualize range image channels
In the Waymo Open dataset, lidar data is stored as a range image. Therefore, this task is about extracting two of the data channels within the range image, which are "range" and "intensity", and convert the floating-point data to an 8-bit integer value range.

**Task preparations**
In file `loop_over_dataset.py`, set the attributes for code execution in the following way:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [0, 1]
    exec_detection = []
    exec_tracking = []
    exec_visualization = ['show_range_image']

#### Result:
<img src= "img/range_image.png"/>

### 1.2 Visualize point-cloud
The goal of this task is to use the Open3D library to display the lidar point-cloud in a 3d viewer in order to develop a feel for the nature of lidar point-clouds.

**Task preparations**
In file loop_over_dataset.py, set the attributes for code execution in the following way:

    data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord'
    show_only_frames = [0, 200]
    exec_detection = []
    exec_tracking = []
    exec_visualization = ['show_pcl']

#### Result:
<img src= "img/pcl_1_f.png"/>
<img src= "img/pcl_2_f.png"/>
<img src= "img/pcl_3_f.png"/>
<img src= "img/pcl_4_f.png"/>
<img src= "img/pcl_5_f.png"/>
<img src= "img/pcl_6_f.png"/>
<img src= "img/pcl_7_f.png"/>
<img src= "img/pcl_8_f.png"/>
<img src= "img/pcl_9_f.png"/>
<img src= "img/pcl_10_f.png"/>


The lidar point cloud sparsely shows the chassis of cars as the identifiable feature. 
The below images are set in different angles to find further features which might be useful for object detection. 

## Step 2: Create Birds-Eye View from Lidar PCL

### 2.1 Convert sensor coordinates to bev-map coordinates (ID_S2_EX1)
The goal of this task is to perform the first step in creating a birds-eye view (BEV) perspective of the lidar point-cloud. Based on the (x,y)-coordinates in sensor space, you must compute the respective coordinates within the BEV coordinate space so that in subsequent tasks, the actual BEV map can be filled with lidar data from the point-cloud.

**Task preparations**
In file loop_over_dataset.py, set the attributes for code execution in the following way:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [0, 1]
    exec_detection = ['bev_from_pcl']
    exec_tracking = []
    exec_visualization = []

#### Result
<img src= "img/vis_bev_co-ordinates.png"/>

### 2.2 Compute intensity layer of bev-map (ID_S2_EX2)
Main goal of this task is to fill the "intensity" channel of the BEV map with data from the point-cloud. In order to do so, we would need to identify all points with the same (x,y)-coordinates within the BEV map and then assign the intensity value of the top-most lidar point to the respective BEV pixel. We could name the resulting list of points `lidar_pcl_top` as this will be re-used later. Also, we will need to normalize the resulting intensity image using percentiles, to make sure that the influence of outlier values are sufficiently mitigated and objects of interest are clearly separated from the background.

**Task preparations**
In file loop_over_dataset.py, set the attributes for code execution in the following way:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [0, 1]
    exec_detection = ['bev_from_pcl']
    exec_tracking = []
    exec_visualization = []

#### Result
<img src= "img/Intensity_map.png"/>

### 2.3 Compute height layer of bev-map (ID_S2_EX3)
In this task the goal was to fill the "height" channel of the BEV map with data from the point-cloud. One could make use of the sorted and pruned point-cloud `lidar_pcl_top` from the previous task and normalize the height in each BEV map pixel by the difference between max. and min. height which is defined in the `configs` structure.

**Task preparations**
In file loop_over_dataset.py, set the attributes for code execution in the following way:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [0, 1]
    exec_data = ['pcl_from_rangeimage']
    exec_detection = ['bev_from_pcl']
    exec_tracking = []
    exec_visualization = []

#### Result
<img src= "img/height_map_pcl.png"/>

## Step 3: 3D Object Detection in BEV Image using pretrained model

### 3.1. Add a second model from a GitHub repo
The model-based detection of objects in lidar point-clouds using deep-learning is a heavily researched area with new approaches appearing in the literature and on GitHub every few weeks. On the website [Papers With Code](https://paperswithcode.com/) and on GitHub, several repositories with code for object detection can be found, such as [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://paperswithcode.com/paper/complex-yolo-real-time-3d-object-detection-on) and [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D).

The goal of this task is to illustrate how a new model can be integrated into an existing framework.

**Task preparations**
In file loop_over_dataset.py, set the attributes for code execution in the following way:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [50, 51]
    exec_detection = ['bev_from_pcl', 'detect_objects']
    exec_tracking = []
    exec_visualization = ['show_objects_in_bev_labels_in_camera']
    configs_det = det.load_configs(model_name="fpn_resnet")

#### Result
Checkout the local variable "detections" in the function `detect_objects` --> from the [code](/student/objdet_detect.py)

### 3.2. Extract 3D bounding boxes from model response
As the model input is a three-channel BEV map, the detected objects will be returned with coordinates and properties in the BEV coordinate space. Therefore, before the detections can move along in the processing pipeline, we need to convert them into metric coordinates --> vehicle space, so that all detections have the format [1, x, y, z, h, w, l, yaw], where 1 denotes the class id for the object type vehicle.

**Task preparations**
In file loop_over_dataset.py, set the attributes for code execution in the following way:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [50, 51]
    exec_detection = ['bev_from_pcl', 'detect_objects']
    exec_tracking = []
    exec_visualization = ['show_objects_in_bev_labels_in_camera']
    model = "fpn_resnet"
    sequence = "3"
    configs_det = det.load_configs(model_name=model)

#### Result
<img src= "img/label_vs_detected.png"/>

## Step 4: Performance Evaluation for Object Detection

### 4.1 Compute intersection-over-union between labels and detections
In this task we aksed to find pairings between ground-truth labels and detections, so that we can determine wether an object has been (a) missed (false negative), (b) successfully detected (true positive) or (c) has been falsely reported (false positive). Based on the labels within the Waymo Open Dataset, our task is to compute the geometrical overlap between the bounding boxes of labels and detected objects and determine the percentage of this overlap in relation to the area of the bounding boxes. A default method in the literature to arrive at this value is called intersection over union.

**Task preparations**
In file loop_over_dataset.py, set the attributes for code execution in the following way:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [50, 51]
    exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
    exec_tracking = []
    exec_visualization = ['show_detection_performance']
    model = "darknet"
    sequence = "1"
    configs_det = det.load_configs(model_name=model)

#### Result
checkout values of variables ---->  `ious` and `center_devs` from the [code](/student/objdet_eval.py)

### 4.2 Compute false-negatives and false-positives
Based on the pairings between gt labels and detected objects, the goal of this task is to determine the number of false positives and false negatives for the current frame. After frames have been processed, an overall performance measure would be computed based on the results.

#### Result
checkout values of variables  ---> `det_performance` from the [code](/student/objdet_eval.py)

### 4.3 Compute precision and recall
After processing all the frames of a sequence, the performance of the object detection algorithm shall now be evaluated. To do so in a meaningful way, the two standard measures "precision" and "recall" will be used, which are based on the accumulated number of positives and negatives from all frames.

**Task preparations**
In file loop_over_dataset.py, set the attributes for code execution in the following way:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [50, 150]
    exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
    exec_tracking = []
    exec_visualization = ['show_detection_performance']
    model = "darknet"
    sequence = "1"
    configs_det = det.load_configs(model_name=model)

#### Result

`precision = 0.9506578947368421, recall = 0.9444444444444444`

<img src= "img/Figure_2.png"/>

To make sure that the code produces plausbile results, the flag `configs_det.use_labels_as_objects` should be set to `True` in a second run.

<img src= "img/Figure_3.png"/>

# Writeup: Track 3D-Objects Over Time

The final project consists of four main steps:

- Step 1: Implement an extended Kalman filter.
- Step 2: Implement track management including track state and track score, track initialization and deletion.
- Step 3: Implement single nearest neighbour data association and gating.
- Step 4: Apply sensor fusion by implementing the nonlinear camera measurement model and a sensor visibility check.

Since the final project is based on the mid-term project, same code files are used as in the mid-term.


## Question Session

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve?

#### Step 1: Extended Kalman Filter (EKF)

To implement an Extended Kalman Filter (EFK), the 6 dimensional system states [x, y, z, vx, vy, vz], process noise Q for constant velocity motion model are designed. To update the current state calculation, h(x) and the Jacobian H are calculated. In the first step, the EKF is applied to simple single-target scenario with lidar only. This is shown in the below figure.

<img src= "img/Figure_1.png"/>

To evaluate the tracking performance, the mean of root mean squared error (RMSE) is calculated. As depicted, the mean RMSE results are smaller than the given reference value 0.35. 

#### Step 2: Track Management
In the second step, the track management is implemented to initialize and delete tracks, set a track state and a track score. First, the track is initialized with unassigned lidar calculation and decrease the track score for unassigned tracks. Then tracks are deleted if the score is too low or P is too big. The below figure shows the single track without track losses in between.

<img src= "img/Figure_rmse_2.png"/>

#### Step 3: Data Association
In the data association, the closest neighbor association matches measurements to tracks and decides which track to update with which measurement by using the Mahalanobis Distance (MHD). The MHD measures the distance between a point vs a distribution. The MHD is used to exclude unlikely track pairs by comparing threshold which is created based on the chi-squared distribution. The following figures show the data association results and corresponding rmse results.

<img src= "img/Figure_9.png"/>
<img src= "img/Figure_rmse_3.png"/>

#### Step 4: Sensor Fusion
To complete the sensor fusion system, the camera measurements including covariance matrix R are implemented. Therby, the function of ``Sensor`` class called ``in_fov()`` is implemented to check if the input state vector of an object can be seen by this sensor. Here, we have to transform from vehicle to sensor coordinates. Furthermore, a nonlinear camera measurement model h(x) is implemented.

First, I set ``configs_det.use_labels_as_objects`` to `Flase` to see how the tracking performance looks with model-based detection. As shown in the figure with rmse results, only few vehicles are tracked correctly. 

<img src= "img/Figure_6.png"/>
<img src= "img/Figure_rmse_6.png"/>

Then, I set ``configs_det.use_labels_as_objects`` to True to see the difference of tracking performance by adding ground truth information. As shown in the below gif, we can see that more vehicles are tracked correctly.

<img src= "img/Figure_9.png"/>
<img src= "img/Figure_rmse_4.png"/>

#### Which part of the project was most difficult for you to complete, and why?
Lectures of EKF, Multi-Target Tracking provide a nice guide and exercises in implementation of EKF, track management, data association and sensor fusion. Personally, for me the sensor fusion part was bit difficult, especially camera measuring model. The projection of 3d space to 2d space was error-prone such that I had to deal with multiple implementation errors. In addition, even the entire lectures and project description are well guided, sometimes I was uncertain about my results of step 1 and 2 which show much better results than in description. I thought I made some mistake. 

### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 
Theoretically, the tracking performance can be stabilized by fusion of multiple sensor. In camera-lidar fusion, camera provides visual information which may show advantages in classification of objects by including color, contrast and brightness. On the other hand, lidar can provide rich spatial information. Combining the two sensor strengths, better tracking performance can be achieved.

My results, as shown in the step 4, a lot of efforts be spent to improve the detection performances such that better tracking performance can be achieved. Even adding ground truth information in detection phase, significant increase of tracking performance can be expected. 

### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
- Since camera is a passive environmental sensor, the detection accuracy might be influenced by lighting conditions. This will lead to decrease of tracking performance.
- When many objects come too close, the occlusion can occur that often leads to wrong tracking of occluded objects.

### 4. Can you think of ways to improve your tracking results in the future?
- Advanced design of process noise covariance might be a conceivable solution to achieve better tracking performance.
- Instead of using `fpn_resnet`, different pretrained neural networks can be used to compare the tracking performances. 
