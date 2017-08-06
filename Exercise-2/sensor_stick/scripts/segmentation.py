#!/usr/bin/env python

# Import modules
import pcl
from pcl_helper import *

# Define functions as required
def vox_downsample(pcl_cloud):
    vox = pcl_cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.0042
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    return cloud_filtered
    
def passthrough_filter(pcl_cloud):
    passthrough = pcl_cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.8
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    passthrough = cloud_filtered..make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -2.75
    axis_max = -1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filteredy = passthrough.filter()
    
    return cloud_filteredy

def ransac_segmentation(pcl_cloud):
    seg = pcl_cloud.make_segmenter()

    # Extract inliers and outliers
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    outlier_filter = pcl_cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()
    
    inliers, coefficients = seg.segment()
    cloud_table   = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True )

    return cloud_table, cloud_objects
    
def euclid_cluster(pcl_cloud):
    white_cloud = XYZRGB_to_XYZ(pcl_cloud) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(600)
    ec.set_MaxClusterSize(2500)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

	#Assign a color corresponding to each segmented object in scene
	cluster_color = get_color_list(len(cluster_indices))

	color_cluster_point_list = []

	for j, indices in enumerate(cluster_indices):
		for i, indice in enumerate(indices):
			color_cluster_point_list.append([
                                            white_cloud[indice][0],
											white_cloud[indice][1],
											white_cloud[indice][2],
											rgb_to_float( cluster_color[j] )
                                           ])

	#Create new cloud containing all clusters, each with unique color
	cluster_cloud = pcl.PointCloud_PointXYZRGB()
	cluster_cloud.from_list(color_cluster_point_list)

    return cluster_cloud

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl( pcl_msg )

    # Voxel Grid Downsampling
    cloud_vox_filtered = vox_downsample( cloud )

    # PassThrough Filter
    cloud_pt_filtered = passthrough_filter( cloud_vox_filtered )

    # RANSAC Plane Segmentation
    cloud_objects, cloud_table = ransac_segmentation( cloud_pt_filtered )

    # Euclidean Clustering
    cluster_cloud = euclid_cluster( cloud_objects )

    # Convert PCL data to ROS messages
    ros_cloud_objects =  pcl_to_ros( cloud_objects )
    ros_cloud_table   =  pcl_to_ros(  cloud_table  )   
	ros_cluster_cloud =  pcl_to_ros( cluster_cloud )

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
	pcl_cluster_pub.publish(ros_cluster_cloud)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber( "/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1 )

    # Create Publishers
    pcl_objects_pub = rospy.Publisher( "/pcl_objects", PointCloud2, queue, queue_size=1 )
    pcl_table_pub   = rospy.Publisher( "/pcl_table",   PointCloud2, queue, queue_size=1 )
	pcl_cluster_pub = rospy.Publisher( "/pcl_cluster", PointCloud2, queue, queue_size=1 )

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()
