import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

# Download the model you want here :
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
PATH_TO_CKPT = 'exported_model/mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'exported_model/all_label_map.pbtxt'

# COCO dataset has 90 categories
NUM_CLASSES = 90

# Manage label_map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Size of the video stream
global_width = 0
global_height = 0

# Init the video stream (with the first plugged webcam)
cap = cv2.VideoCapture(0)
if cap.isOpened():
    # get vcap property
    global_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    global_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Init TF Graph and get all needed tensors
detection_graph = tf.Graph()
with detection_graph.as_default():
    # Init the graph
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # Get all tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    # detection_masks tensor need ops
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, global_height, global_width)
    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


# Detect objects in a given image and return the image with detected objects
def detect_objects(image_np, sess):
    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    # Display boxes and color pixels
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True)

    return image_np


# Run the program
if __name__ == '__main__':
    # Do that here and save a lot of time
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                # Get the last frame
                frame = cap.read()[1]
                # Process last img
                new_frame = detect_objects(image_np, sess)
                # Display the resulting frame
                cv2.imshow('new_frame', new_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
