""" the InferenceYOLOv5 class implementation. """
import cv2
import numpy as np
import onnxruntime as rt


CONF_THRESHOLD    = 0.30
IOU_NMS_THRESHOLD = 0.75
IOS_NMS_THRESHOLD = 0.90
IMG_SIZE          = 2048

WINDOW_SIZE    = 3100
WINDOW_OVERLAP = 600
WINDOW_STEP    = WINDOW_SIZE - WINDOW_OVERLAP
OFFSET         = 25


class InferenceYOLOv5:
    """ Wraps a YOLOv5 model to run inference on images.

    Parameters
    ----------
    onnx_path: str
        A path to an ONNX file containing YOLOv5.

    Attributes
    ----------
    _session: onnxruntime.InferenceSession
        The loaded model.
    """
    def __init__(self, onnx_path: str):
        self._session = rt.InferenceSession(onnx_path, None)

    # ------------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------------
    def inference(self, input_image: np.ndarray) -> list:
        """ Runs inference on the specified image.

        Parameters
        ----------
        image: np.ndarray
            An 1-channel grayscale image.

        Returns
        -------
        list
            A list of detected bounding boxes in this format:
            [x1, x2, y1, y2, conf, class1, class2]
        """
        image = self._format_image_to_yolov5(input_image)
        input_name = self._session.get_inputs()[0].name
        pred = self._session.run([], {input_name: image})[0][0]
        conf_pred = self._process_inference_results(pred, input_image.shape)

        return conf_pred

    def inference_big_image(self, input_image: np.ndarray) -> list:
        """ Runs inference on a big image via sliding window algorithm.

        Parameters
        ----------
        input_image: np.ndarray
            The specified big image.

        Returns
        -------
        list
            A list of predicted bounding boxes.
        """

        input_image_size = input_image.shape  # 1-channel grayscale image

        # How many steps sliding window should take
        grid_x = input_image_size[1] // (WINDOW_SIZE - WINDOW_OVERLAP) + 1
        grid_y = input_image_size[0] // (WINDOW_SIZE - WINDOW_OVERLAP) + 1

        # All predicted bounding boxes go here
        predictions = []

        # Making predictions
        for i in range(grid_y):
            for j in range(grid_x):
                # ------------------------------------------------------------
                # Take an image crop
                # ------------------------------------------------------------
                start_y = i * WINDOW_STEP
                start_x = j * WINDOW_STEP
                end_y   = start_y + WINDOW_SIZE
                end_x   = start_x + WINDOW_SIZE

                if end_x > input_image_size[1]:
                    end_x = input_image_size[1]

                if end_y > input_image_size[0]:
                    end_y = input_image_size[0]

                # Make bg image for crops
                current_crop = np.ones((WINDOW_SIZE, WINDOW_SIZE),
                                       dtype=np.uint8) * 255

                current_crop[0:end_y-start_y, 0:end_x-start_x] =\
                    input_image[start_y:end_y, start_x:end_x]

                # ------------------------------------------------------------
                # Make predictions for one crop
                # ------------------------------------------------------------
                results = self.inference(current_crop)

                # Local coordinates to global coordinates
                for box in results:
                    box[0] += start_x
                    box[1] += start_y
                    box[2] += start_x
                    box[3] += start_y

                predictions.append(results)

        # Removing dublicated bounding boxes in window overlap areas
        for i in range(grid_y):
            for j in range(grid_x):
                current_crop = predictions[i * grid_x + j]

                # ============================================================
                # X AXIS OVERLAPPING
                # ============================================================
                if j < grid_x - 1:
                    next_crop = predictions[i * grid_x + (j + 1)]
                    boxes_to_delete = []

                    # --------------------------------------------------------
                    # Delete all boxes on borders of window overlap areas
                    # --------------------------------------------------------
                    # Right border of overlap area
                    for box in current_crop:
                        if box[2] > j * WINDOW_STEP + WINDOW_SIZE - OFFSET:
                            boxes_to_delete.append(box)

                    for box in boxes_to_delete:
                        current_crop.remove(box)

                    boxes_to_delete.clear()

                    # Left border of overlap area
                    for box in next_crop:
                        if box[0] < (j + 1) * WINDOW_STEP + OFFSET:
                            boxes_to_delete.append(box)

                    for box in boxes_to_delete:
                        next_crop.remove(box)

                    boxes_to_delete.clear()

                    # --------------------------------------------------------
                    # Delete dublicates in the overlaping area volume
                    # --------------------------------------------------------
                    for box in current_crop:
                        if box[0] >= (j + 1) * WINDOW_STEP - OFFSET:
                            for pair in next_crop:
                                if pair[2] <= j * WINDOW_STEP + WINDOW_SIZE + OFFSET and\
                                        self._iou(box, pair) >= 0.90:
                                    boxes_to_delete.append(pair)

                    for box in boxes_to_delete:
                        next_crop.remove(box)

                    boxes_to_delete.clear()

                    # --------------------------------------------------------
                    # Delete in the diagonal overlaping area volume
                    # --------------------------------------------------------
                    if i < grid_y - 1:
                        next_crop = predictions[(i + 1) * grid_x + (j + 1)]

                        for box in current_crop:
                            if box[0] >= (j + 1) * WINDOW_STEP and\
                                    box[1] >= (i + 1) * WINDOW_STEP:
                                for pair in next_crop:
                                    if pair[2] <= j * WINDOW_STEP + WINDOW_SIZE and\
                                            pair[3] <= i * WINDOW_STEP + WINDOW_SIZE and\
                                            self._iou(box, pair) >= 0.90:
                                        boxes_to_delete.append(pair)

                        for box in boxes_to_delete:
                            next_crop.remove(box)

                        boxes_to_delete.clear()

                # ============================================================
                # Y AXIS OVERLAPPING
                # ============================================================
                if i < grid_y - 1:
                    next_crop = predictions[(i + 1) * grid_x + j]

                    # --------------------------------------------------------
                    # Delete all boxes on borders of window overlap areas
                    # --------------------------------------------------------
                    # Top border of overlap area
                    for box in current_crop:
                        if box[3] > i * WINDOW_STEP + WINDOW_SIZE - OFFSET:
                            boxes_to_delete.append(box)

                    for box in boxes_to_delete:
                        current_crop.remove(box)

                    boxes_to_delete.clear()

                    # Bottom border of overlap area
                    for box in next_crop:
                        if box[1] < (i + 1) * WINDOW_STEP + OFFSET:
                            boxes_to_delete.append(box)

                    for box in boxes_to_delete:
                        next_crop.remove(box)

                    boxes_to_delete.clear()

                    # --------------------------------------------------------
                    # Delete dublicates in the overlaping area volume
                    # --------------------------------------------------------
                    for box in current_crop:
                        if box[1] >= (i + 1) * WINDOW_STEP - OFFSET:
                            for pair in next_crop:
                                if pair[3] <= i * WINDOW_STEP + WINDOW_SIZE + OFFSET and\
                                        self._iou(box, pair) >= 0.90:
                                    boxes_to_delete.append(pair)

                    for box in boxes_to_delete:
                        next_crop.remove(box)

                    boxes_to_delete.clear()

        return predictions

    # ------------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------------
    def _format_image_to_yolov5(self, input_image: np.ndarray) -> np.ndarray:
        """ Converts images to pass them to the YOLOv5 model.

        Note
        ----
        Pipeline:
            1) Resizing to the expected size;
            2) Stacking 3 channel image;
            2) Converting from [h, w, c] to [c, h, w];
            3) Mapping pixel values from 0...255 to 0...1.

        Parameters
        ----------
        input_image: np.ndarray
            An 1-channel grayscale image.

        Returns
        -------
        np.ndarray
            An image in format expected by YOLOv5 models.
        """
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

        # Adding padding to image results in incorrect regression!

        image = cv2.resize(rgb_image, (IMG_SIZE, IMG_SIZE),
                           interpolation=cv2.INTER_LINEAR)

        image = image.transpose(2, 0, 1)

        image = image.astype(np.float32)
        image /= 255.0

        image = image.reshape(1, *image.shape)

        return image

    def _change_bbox_format(self, bbox: list) -> list:
        """ Changes the format of the specified bounding box.

        Parameters
        ----------
        bbox: list
            An old bounding box in this format:
            [x_center, y_center, w, h, ...](floats)

        Returns
        -------
        list
            A new bounding box in this format:
            [x, y1, x2, y2, ...](floats)
        """
        x1 = bbox[0] - bbox[2] / 2
        y1 = bbox[1] - bbox[3] / 2

        x2 = bbox[0] + bbox[2] / 2
        y2 = bbox[1] + bbox[3] / 2

        return [x1, y1, x2, y2, *bbox[4:]]

    def _clip_bboxes(self, bboxes: list, image_shape: tuple) -> None:
        """ Clips bounding boxes along the borders of an image in-place.

        Parameters
        ----------
        bboxes: list
            A list of detected bounding boxes.
        image_shape: tuple
            The width and height of an image.
        """
        for box in bboxes:
            if box[0] < 0: box[0] = 0
            if box[1] < 0: box[1] = 0
            if box[2] >= image_shape[1]: box[2] = image_shape[1] - 1
            if box[3] >= image_shape[0]: box[3] = image_shape[0] - 1

    def _ios(self, box1: list, box2: list) -> float:
        """ Calculates the value of the Intersection over Smallest metric.

        Note
        ----
        The formula:

            IOS = intersection_area / smallest_bbox_area

        This metric allows to avoid situations when 2 non-dividing cells
        are found inside a dividing cell in telophase.

        Parameters
        ----------
        box1 : list
            The first bounding box.
        box2 : list
            The second bounding box.

        Returns
        -------
        float
            The IOS value.
        """
        xa = max(box1[0], box2[0])
        ya = max(box1[1], box2[1])
        xb = min(box1[2], box2[2])
        yb = min(box1[3], box2[3])

        intersection_area = max(0, xb - xa) * max(0, yb - ya)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        smallest_area = box1_area if box1_area < box2_area else box2_area

        return intersection_area / smallest_area

    def _iou(self, box1: list, box2: list) -> float:
        """ Calculates the value of the Intersection over Union metric.

        Parameters
        ----------
        box1 : list
            The first bounding box.
        box2 : list
            The second bounding box.

        Returns
        -------
        float
            The IOU value.
        """
        xa = max(box1[0], box2[0])
        ya = max(box1[1], box2[1])
        xb = min(box1[2], box2[2])
        yb = min(box1[3], box2[3])

        intersection_area = max(0, xb - xa) * max(0, yb - ya)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection_area / (box1_area + box2_area - intersection_area)

    def _nms(self, bboxes: list) -> list:
        """ Filters detected objects with non-maximum suppression.

        Parameters
        ----------
        bboxes: list
            A list of bounding boxes.

        Returns
        -------
        list
            A filtered list of bounding boxes.
        """
        nms_results = []

        while len(bboxes) != 0:
            boxes_to_delete = []
            max_conf_box = max(bboxes, key=lambda box: box[4])
            nms_results.append(max_conf_box)
            bboxes.remove(max_conf_box)

            for box in bboxes:
                if self._iou(max_conf_box, box) > IOU_NMS_THRESHOLD or\
                        self._ios(max_conf_box, box) > IOS_NMS_THRESHOLD:
                    boxes_to_delete.append(box)

            for box in boxes_to_delete:
                bboxes.remove(box)

        return nms_results

    def _process_inference_results(self, results: list,
                                   image_shape: tuple) -> list:
        """ Handles detected bounding boxes after inference.

        Note
        ----
        1) Changes the bounding box format;
        2) Filters boxes by confidence level;
        3) Rescales boxes to the original image;
        4) Clips boxes to the original image limits;
        5) Applies Non-Maximum Suppression.

        Parameters
        ----------
        results: list
            A list of detected bounding boxes immediately after inference.

        Returns
        -------
        list
            A filtred list of bounding boxes.
        """
        conf_results = [self._change_bbox_format(x) for x in results
                        if x[4] > CONF_THRESHOLD]
        self._rescale_bboxes(conf_results, image_shape)
        self._clip_bboxes(conf_results, image_shape)
        conf_results = self._nms(conf_results)

        return conf_results

    def _rescale_bboxes(self, bboxes: list, image_shape: tuple) -> None:
        """ Scales detected boxes for the original image in-place.

        Parameters
        ----------
        bboxes: list
            A list of detected bounding boxes.
        image_shape: tuple
            The width and height of an image.
        """
        x_factor = image_shape[1] / IMG_SIZE
        y_factor = image_shape[0] / IMG_SIZE

        for box in bboxes:
            box[0] = int(round(box[0] * x_factor, 0))
            box[1] = int(round(box[1] * y_factor, 0))
            box[2] = int(round(box[2] * x_factor, 0))
            box[3] = int(round(box[3] * y_factor, 0))


if __name__ == '__main__':
    # ------------------------------------------------------------------------
    # Run inference on a test image
    # ------------------------------------------------------------------------
    model = InferenceYOLOv5('yolov5_allium.onnx')
    test_image = cv2.imread('Nd_Control_1_1__3-4.jpg', cv2.IMREAD_GRAYSCALE)
    pred = model.inference(test_image)

    #   x1   y1   x2   y2   conf   div    ndiv
    # [int, int, int, int, float, float, float]

    # ------------------------------------------------------------------------
    # Check results by drawing
    # ------------------------------------------------------------------------
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

    for box in pred:
        color = (255, 255, 0) if box[5] > box[6] else (0, 255, 0)
        cv2.rectangle(test_image,
                      (box[0], box[1]),
                      (box[2], box[3]), color, 10)

    test_image = cv2.resize(test_image,
                            (test_image.shape[0] // 5,
                             test_image.shape[1] // 5),
                            interpolation=cv2.INTER_LINEAR)
    cv2.imshow(f'{len(pred)}', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
