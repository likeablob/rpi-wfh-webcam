#! /usr/bin/env python3
import os
import sys
import time
import queue
import argparse
import cv2
import pafy
import numpy as np
import tensorflow as tf
from threading import Thread, Event
from logging import basicConfig, getLogger
from PIL import Image

# Set up logger
basicConfig()
logger = getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


class BodyPixEstimator():
    def __init__(self, model_path, output_stride=16, default_shrink_factor=2):
        self.output_stride = output_stride
        self.default_shrink_factor = default_shrink_factor

        self._load_model(model_path)

    def _load_model(self, model_path):
        # Load the model
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map=None,
                                return_elements=None,
                                name=""
                                )
        self.graph = graph

    def preprocess_input_image(self, orig_image, shrink_factor=None):
        image = Image.fromarray(orig_image)
        image_width, image_height = image.size

        factor = shrink_factor or self.default_shrink_factor

        # Calculate the size of input image tensor
        input_width = int(image_width / factor) // self.output_stride\
            * self.output_stride + 1
        input_height = int(image_height / factor) // self.output_stride\
            * self.output_stride + 1
        logger.debug(f'input res: {input_width} x {input_height}')

        # Resize the original image
        image = image.resize((input_width, input_height))
        image_arr = tf.keras.preprocessing.image.img_to_array(image,
                                                              dtype=np.float32)

        # Calculate the size of output tensor
        output_width = int((input_width - 1) / self.output_stride) + 1
        output_height = int((input_height - 1) / self.output_stride) + 1
        logger.debug(f'output res: {output_width} x {output_height}')

        # Normalize the pixels (for Resnet): TODO: Parametrize this
        # m = np.array([-123.15, -115.90, -103.06])
        # image_arr = np.add(image_arr, m)

        # Normalize the pixels (for MobileNet)
        # See https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/mobilenet.ts#L25 # noqa: E501
        image_arr = image_arr/127.5 - 1.0

        return image_arr[tf.newaxis, ...]

    def estimate(self, input_data):
        with tf.compat.v1.Session(graph=self.graph) as sess:
            # FIXME: Replace with tfjs.util.get_input_tensors(graph)
            input_tensor_names = ['sub_2:0']
            # FIXME: Replace with tfjs.util.get_output_tensors(graph)
            output_tensor_names = ['float_short_offsets:0',
                                   'float_segments:0',
                                   'float_part_heatmaps:0',
                                   'float_long_offsets:0',
                                   'float_heatmaps:0',
                                   'MobilenetV1/displacement_fwd_2/BiasAdd:0',
                                   'MobilenetV1/displacement_bwd_2/BiasAdd:0',
                                   'float_part_offsets:0']

            input_tensor = self.graph.get_tensor_by_name(input_tensor_names[0])
            results = sess.run(output_tensor_names, feed_dict={
                input_tensor: input_data})
            results = dict(zip(output_tensor_names, results))

        # Canonical output labels, sorted for the later use
        output_labels = sorted([
            "short_offsets",
            "displacement_bwd",
            "displacement_fwd",
            "heatmaps",
            "long_offsets",
            "part_heatmaps",
            "segments",
            "part_offsets",
        ], key=len, reverse=True)

        labeled_results = {}
        for label in output_labels:
            key = next(
                (x for x in output_tensor_names if x.find(label) != -1), None)
            # Prevent the same key being selected again
            output_tensor_names.remove(key)
            # Squeeze the batch axis [1, H, W, C] => [H, W, C]
            result = np.squeeze(results[key], 0)
            labeled_results[label] = result

        return labeled_results

    def generate_mask(self, segments, threshold=0.7):
        scores = tf.sigmoid(segments)
        mask = tf.math.greater(
            scores, tf.constant(threshold))
        mask = tf.dtypes.cast(mask, tf.int32)
        mask = np.squeeze(mask.numpy(), -1)
        return mask


def mask_image_generator(is_running, estimator, input_q, output_q):
    while is_running.is_set():
        image = input_q.get()
        image_height, image_width, _ = image.shape

        # Perform preprocess
        start = time.time()
        input_data = estimator.preprocess_input_image(image)
        duration = time.time() - start
        logger.debug(f"Preprocess took {duration: .3f} s")

        # Run the estimator
        start = time.time()
        result = estimator.estimate(input_data)
        duration = time.time() - start
        logger.debug(
            f"Estimation took {duration: .3f} s ({1/duration: .3f} fps)")

        # Generate the mask image to composite
        mask = estimator.generate_mask(result["segments"])
        mask_img = Image.fromarray(mask * 255)
        mask_img = mask_img.resize(
            (image_width, image_height), Image.BICUBIC).convert("RGB")
        mask_img = tf.keras.preprocessing.image.img_to_array(
            mask_img, dtype=np.uint8)

        # Add to the queue
        if output_q.full():
            output_q.get()
        output_q.put_nowait(mask_img)


def main(ARGS):
    # Init variables
    latest_mask_img = None
    input_q = queue.Queue(maxsize=1)
    output_q = queue.Queue(maxsize=1)
    estimator = BodyPixEstimator(ARGS.model, ARGS.stride, ARGS.shrink_factor)

    # Open the capture device
    cap = cv2.VideoCapture(ARGS.cap_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ARGS.cap_res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARGS.cap_res[1])

    if not cap.isOpened():
        logger.error("Could not open the specified device.")
        return

    # Load the background image/video
    if ARGS.bg_video:
        cap_file = cv2.VideoCapture(ARGS.bg_video)
    elif ARGS.bg_url:
        vPafy = pafy.new(ARGS.bg_url)
        play = vPafy.getbestvideo()
        cap_file = cv2.VideoCapture(play.url)

    bg_img = cv2.imread(ARGS.bg_image)
    bg_img = cv2.resize(bg_img, dsize=ARGS.cap_res)

    # Start the prediction thread
    is_running = Event()
    is_running.set()

    th = Thread(name="mask_image_generator", target=mask_image_generator,
                args=(is_running, estimator, input_q, output_q,))
    th.isDaemon = True
    th.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warn("cap.read() failed.")
            continue

        # Flip the image if requested
        if ARGS.flip != -1:
            frame = cv2.flip(frame, ARGS.flip)

        # Pass the image to the generator thread
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if input_q.full():
            input_q.get()
        input_q.put(im_rgb, block=False)

        # Get the mask image if possible
        if output_q.qsize() > 0:
            latest_mask_img = output_q.get_nowait()

        # Load a frame from the background video file
        if cap_file:
            ret, bg_img = cap_file.read()
            if not ret:
                cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, bg_img = cap_file.read()

            bg_img = cv2.resize(bg_img, dsize=ARGS.cap_res)

        if latest_mask_img is not None:
            # Composite the foreground and background images
            mask = latest_mask_img
            composed = np.zeros(frame.shape, dtype=frame.dtype)
            for i in range(3):
                composed[:,:,i] = bg_img[:,:,i] * (1-mask[:,:,i]/255) + frame[:,:,i] * (mask[:,:,i]/255)

            # Output the frame
            if ARGS.show_only_gui:
                cv2.imshow("frame", composed)
            else:
                sys.stdout.buffer.write(composed.tobytes())
        else:
            if ARGS.show_only_gui:
                cv2.imshow("frame", bg_img)
            else:
                sys.stdout.buffer.write(bg_img.tobytes())

        if cv2.waitKey(1) != -1:
            break

    is_running.clear()
    th.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Init argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Streaming video to stdout with applying special effects. \
            Use ffmpeg to write the stream to another video device.')
    parser.add_argument('-m', '--model', metavar='PB_FILE',
                        type=str, default="models/face-detection-retail-0004",
                        help='Model base path.')
    parser.add_argument('-s', '--stride',
                        type=int, default=16,
                        help='Stride for the model output.')
    parser.add_argument('-k', '--shrink-factor', metavar='FACTOR',
                        type=float, default=2,
                        help='Shrink factor for the input\
                                (1 for better fitting).')

    parser.add_argument('-c', '--cap-source', metavar='CAMERA_SOURCE',
                        type=int, default=0, help='V4L2 Camera source id.')
    parser.add_argument('-r', '--cap-res', metavar='WH',
                        type=int, nargs=2, default=(640, 480),
                        help='Camera capture resolution.')
    parser.add_argument('-f', '--flip',
                        type=int, default=-1,
                        help='Flip video by cv2.flip(data, N).')

    parser.add_argument('-i', '--bg-image', metavar='IMAGE_FILE',
                        type=str, default="bg.jpg",
                        help='Background image.')
    parser.add_argument('-e', '--bg-video', metavar='VIDEO_FILE',
                        type=str, default=None,
                        help='Background video.')
    parser.add_argument('-u', '--bg-url', metavar='VIDEO_URL',
                        type=str, default=None,
                        help='Background video url.')
    parser.add_argument('-g', '--show-only-gui',
                        action='store_true',
                        help='Show results visually (needs X11).')

    ARGS = parser.parse_args()
    main(ARGS)
