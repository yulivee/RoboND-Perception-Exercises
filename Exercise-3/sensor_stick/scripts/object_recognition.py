#!/usr/bin/env python

import numpy as np
from segmentation import *

import pickle

from visualization_msgs.msg import Marker
from sensor_stick.pcl_helper import *

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
    cluster_indices, white_cloud = euclid_cluster( cloud_objects )

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = cluster_mask( cluster_indices, white_cloud )

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects =  pcl_to_ros( cloud_objects )
    ros_cloud_table   =  pcl_to_ros(  cloud_table  )
    ros_cluster_cloud =  pcl_to_ros( cluster_cloud )

    # Exercise-3 TODOs: 
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects, detected_objects_labels = classify_cluster(cluster_indices, white_cloud, cloud_objects, clf, encoder, scaler, object_markers_pub)


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
