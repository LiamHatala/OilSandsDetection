import torch
from torchvision import models, transforms
from torch.serialization import add_safe_globals
from torch._dynamo.eval_frame import OptimizedModule
import cv2
import numpy as np
from PIL import Image


def get_label(lvl):
    if lvl < 55:
        label = "LOW"
    elif lvl < 80:
        label = "ADEQUATE"
    else:
        label = "HIGH"

    return label


def get_pretrained_model(mdl_pth: str):
    """Helper function: returns model loaded with pretrained weights"""
    model = torch.load(mdl_pth, weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model to the right device (GPU or CPU)
    model = model.to(device)
    return model


def show_model_output_video(model_path, video_path):
    """Generate video of model prediction"""
    add_safe_globals([OptimizedModule])
    model = get_pretrained_model(model_path)
    model.eval()
    display_video(video_path, model)


def get_level(roi_img, prev_lvl=None):
    """
    Take vertical gradient over probabilities averaged horizontally,
    the level should be where there is maximum deviation.
    """
    roi_img = np.where(roi_img > 0.49, 1, 0)
    img_8u = np.asarray(roi_img, dtype=np.uint8)
    img_8u = cv2.morphologyEx(img_8u, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, hierarchy = cv2.findContours(img_8u, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    img_outline = np.copy(img_8u)
    img_outline = cv2.drawContours(img_outline, [contours[0]], -1, (255, 0, 0), 2)

    lvl_top = np.nonzero(img_outline)
    contour_points = lvl_top
    lower_q = len(lvl_top[0]) // 5
    lvl_bot = int(
        np.median(lvl_top[0][lower_q : min(2 * lower_q, lvl_top[0].shape[0])])
    )
    lvl_top = int(np.median(lvl_top[0][:lower_q]))
    level = (lvl_top + lvl_bot) / 2

    std = np.std(contour_points[0]) # get the standard deviation of the contour points y coordinates
    # get the y coordinate for top

    y_top = int(np.mean(contour_points[0])) + std

    y_bot = int(np.mean(contour_points[0])) - std

    if prev_lvl is not None:
        level = 1.0 * level + 0.0 * prev_lvl

    return level, lvl_top, lvl_bot, y_top, y_bot


def get_conf(out: cv2.Mat):
    # Return the confidence interval represented as the variance
    # in the vertical gradients across segmented image, averaged
    # across columns.
    grad = np.std(np.gradient(out, axis=0), axis=1)
    return grad


def display_video(video_path, model):
    level_dict = {0: "Low Level", 1: "Adequate Level", 2: "High Level"}
    cap = cv2.VideoCapture(video_path)
    cap_write = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (1330, 720)
    )
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    paused = False
    frame_index = 0
    cv2.namedWindow("Oil Sands Level Classification")
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_level = None
    x, y, w, h = (350, 0, 500, 720)

    lvl_bar_width = 50
    lvl_bar = np.full((h, lvl_bar_width, 3), (0, 0, 0), dtype=np.uint8)

    def display_frame(frame):
        nonlocal prev_level
        """process frame and display"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        transform = transforms.Compose(
            [
                transforms.Resize((720, 1280)),
                transforms.ToTensor(),
            ]
        )
        img = transform(img).unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img = img.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(img)
        output = output = outputs["out"].squeeze(0)
        out_img = output[0].cpu().numpy()

        predicted_level = get_level(out_img[y : y + h, x : x + w], prev_level)
        if prev_level is None:
            prev_level = predicted_level

        line_h = int(predicted_level)
        top_line = max(int(line_h - 0.15 * h), 0)
        bot_line = min(int(line_h + 0.1 * h), h)

        frame = cv2.line(
            frame,
            (0, top_line),
            (frame.shape[1], top_line),
            (0, 0, 255),
            2,
        )
        frame = cv2.line(
            frame, (0, bot_line), (frame.shape[1], bot_line), (0, 0, 255), 2
        )

        cv2.putText(
            frame,
            f"Level: {((h - line_h) / h) * 100:3.0f}%",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (163, 48, 31),
            2,
        )

        cv2.putText(
            frame,
            f"{get_label((h - top_line) / h * 100)}",
            (50, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (163, 48, 31),
            2,
        )
        print(((h - line_h) / h) * 100)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # concatenate lvl_bar and draw rectangle
        frame = np.concatenate((frame, lvl_bar), axis=1)
        frame = cv2.rectangle(
            frame, (1280 + 35, line_h), (1280 + lvl_bar_width, 720), (163, 31, 48), 15
        )

        cv2.imshow("Oil Sands Level Classification", frame)
        cap_write.write(frame)

    while cap.isOpened():
        if not paused:
            frame_index += 300  # Move forward 30 frames (adjust as needed)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            # Display the frame with predictions
            display_frame(frame)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # Quit
            break
        elif key == ord("p"):  # Pause/Play
            paused = not paused
        elif key == ord("r"):  # Rewind
            frame_index = max(
                frame_index - 30, 0
            )  # Move back 30 frames (adjust as needed)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            paused = False

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_pth = "best_mdl_wts.pt"
    # video_pth = "C39-T3_CRUSHER_HOPPER_2025-01-15_09_00_00_000.mp4"
    video_pth = "C39-T3_CRUSHER_HOPPER_2025-01-10_15_07_25_696.mp4"

    show_model_output_video(model_pth, video_pth)
