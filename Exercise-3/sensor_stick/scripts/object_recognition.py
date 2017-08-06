#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from segmentation import *

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl( pcl_msg )

    # TODO: Voxel Grid Downsampling
    cloud_vox_filtered = vox_downsample( cloud )

    # TODO: PassThrough Filter
    cloud_pt_filtered = passthrough_filter( cloud_vox_filtered )

    # TODO: RANSAC Plane Segmentation
    # TODO: Extract inliers and outliers
    cloud_table, cloud_objects = ransac_segmentation( cloud_pt_filtered )
    

    # TODO: Euclidean Clustering
    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = euclid_cluster( cloud_objects )

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects =  pcl_to_ros( cloud_objects )
    ros_cloud_table   =  pcl_to_ros(  cloud_table  )
    ros_cluster_cloud =  pcl_to_ros( cluster_cloud )

 
    return

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

        # Grab the points for the cluster
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_out.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
	ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # TODO: complete this step just as you did before in capture_features.py
	chists = compute_color_histograms(sample_cloud, using_hsv=True)
        normals = get_normals(ros_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        labeled_features.append([feature, model_name])

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))


        # Compute the associated feature vector

        # Make the prediction

        # Publish a label into RViz

        # Add the detected object to the list of detected objects.

    # Publish the list of detected objects
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber( "/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1 )

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher( "/pcl_objects", PointCloud2, queue_size=1 )
    pcl_table_pub   = rospy.Publisher( "/pcl_table",   PointCloud2, queue_size=1 )
    pcl_cluster_pub = rospy.Publisher( "/pcl_cluster", PointCloud2, queue_size=1 )
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()
