"""ROS2 Humble node: subscribes to camera images, publishes Detection2DArray.

Topic I/O:
  Subscribe: /camera/image_raw          (sensor_msgs/Image)
  Publish:   /perception/detections     (vision_msgs/Detection2DArray)
  Publish:   /perception/det_image      (sensor_msgs/Image)  [debug, optional]

Launch:
  ros2 run urban_det detection_node --ros-args \
    -p engine_path:=/opt/model/rtdetr.engine \
    -p conf_threshold:=0.35 \
    -p publish_debug_image:=true
"""

from __future__ import annotations

# Guard: only importable inside a ROS2 environment
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import (
        BoundingBox2D,
        Detection2D,
        Detection2DArray,
        ObjectHypothesisWithPose,
    )
    from cv_bridge import CvBridge
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]

import numpy as np

from urban_det.deployment.edge.tensorrt_infer import TRTDetector


class DetectionNode(Node):  # type: ignore[misc]
    """ROS2 node wrapping TRTDetector for real-time AV perception."""

    def __init__(self) -> None:
        if not _ROS2_AVAILABLE:
            raise RuntimeError("ROS2 (rclpy) is not installed.")
        super().__init__("urban_detection_node")

        # Parameters
        self.declare_parameter("engine_path", "model.engine")
        self.declare_parameter("conf_threshold", 0.35)
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("input_size", 640)

        engine_path = self.get_parameter("engine_path").value
        conf_thr = self.get_parameter("conf_threshold").value
        self._pub_debug = self.get_parameter("publish_debug_image").value
        size = self.get_parameter("input_size").value

        self.detector = TRTDetector(engine_path, (size, size), conf_thr)
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 10
        )
        self.pub_det = self.create_publisher(Detection2DArray, "/perception/detections", 10)
        if self._pub_debug:
            self.pub_img = self.create_publisher(Image, "/perception/det_image", 10)

        self.get_logger().info(
            f"DetectionNode ready — engine: {engine_path}, conf: {conf_thr}"
        )

    def _image_callback(self, msg: "Image") -> None:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = self.detector.infer(bgr)

        det_array = Detection2DArray()
        det_array.header = msg.header

        for d in detections:
            x1, y1, x2, y2 = d["box"]
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(d["label"])
            hyp.hypothesis.score = d["score"]
            det.results.append(hyp)
            det_array.detections.append(det)

        self.pub_det.publish(det_array)

        if self._pub_debug:
            import cv2
            for d in detections:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(bgr, f"{d['label']} {d['score']:.2f}",
                            (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))

        self.get_logger().debug(
            f"Detected {len(detections)} objects in {self.detector.latency_ms:.1f} ms"
        )


def main(args=None) -> None:
    if not _ROS2_AVAILABLE:
        print("ERROR: rclpy not available. Run inside a ROS2 environment.")
        return
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
