#!/usr/bin/env python3
import argparse
from src.detectors import ColorCubeDetector, YOLODetector
from src.processors import PointCloudProcessor
from src.visualizers import draw_vertices, show_frames
from src.utils import load_config, logger
import cv2

def main(args):
    config = load_config()
    detector = None
    processor = None
    cap = None

    # Set up frame source
    if args.device == 'realsense':
        processor = PointCloudProcessor(config)
        depth_img, color_img, depth_frame, color_frame = processor.get_frames()
    else:  # webcam
        cap = cv2.VideoCapture(0)
        ret, color_img = cap.read()
        if not ret:
            logger.error("Failed to read from webcam")
            return
        depth_img = None
        depth_frame = None
        color_frame = None

    # Initialize detector based on mode
    if args.mode == 'color':
        detector = ColorCubeDetector(config)
    elif args.mode == 'yolo':
        detector = YOLODetector(config)
    else:
        logger.error("Mode must be 'color' or 'yolo'")
        return

    last_pos = None
    while True:
        # Get new frames for this iteration
        if args.device == 'webcam':
            ret, color_img = cap.read()
            if not ret:
                break
            depth_img = None
            depth_frame = None
            color_frame = None
        else:  # realsense
            depth_img, color_img, depth_frame, color_frame = processor.get_frames()

        # Process frame
        if args.mode == 'color':
            verts, contour, mask = detector.detect(color_img)
            if contour is not None:
                vis_frame = draw_vertices(color_img.copy(), verts)
                if processor:
                    pc_verts = processor.mask_pointcloud(color_frame, depth_frame, mask)
                    processor.visualize_pc(pc_verts)
                    show_frames(vis_frame, depth_img, mask)
                    last_pos = np.mean(verts, axis=0) if len(verts) > 0 else last_pos
                else:
                    show_frames(vis_frame, mask=mask)
            elif last_pos is not None:
                logger.info(f"Lost cube. Last pos: {last_pos}")
        elif args.mode == 'yolo':
            bboxes, classes = detector.detect(color_img)
            vis_frame = color_img.copy()
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(vis_frame, str(classes[i]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            show_frames(vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    if processor:
        del processor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealSense Cube Detector')
    parser.add_argument('--mode', choices=['color', 'yolo'], default='color', help='Detection mode')
    parser.add_argument('--device', choices=['realsense', 'webcam'], default='realsense', help='Input device')
    args = parser.parse_args()
    main(args)