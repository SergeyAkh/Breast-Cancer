from tkinter import messagebox

import tkinter as tk
from tkinter import filedialog

import tensorflow

from torchvision import transforms as T
from torchvision.ops import nms

import torchvision.models.detection as detection
import torch
import numpy as np

from PIL import Image, ImageDraw, ImageTk

import warnings
warnings.filterwarnings('ignore')

target_size=(256, 256)
class_names_step_one = ["normal", "cancer"]
class_names_step_two = ["benign", "malignant"]
transform = T.Compose([
    T.ToTensor(),  # Convert image to tensor
    T.Resize(target_size),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


model_step_one = detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model_step_one.roi_heads.box_predictor.cls_score.in_features
model_step_one.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=3)

model_step_one = tensorflow.keras.models.load_model('/Users/sergeiakhmadulin/My Drive/Breast Cancer/model_to_allocate_cancer.keras')

model_step_two = detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model_step_two.roi_heads.box_predictor.cls_score.in_features
model_step_two.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=3)

model_step_two.load_state_dict(torch.load('/Users/sergeiakhmadulin/My Drive/Breast Cancer/cancer_classifier.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_nms(predictions, iou_threshold=0.5, score_threshold=0.3):
    new_predictions = []

    for prediction in predictions:
        boxes_all = prediction['boxes']
        scores_all = prediction['scores']
        labels_all = prediction['labels']

        # Filter out low-score predictions
        keep = scores_all >= score_threshold
        boxes = boxes_all[keep]
        scores = scores_all[keep]
        labels = labels_all[keep]

        # Apply NMS (Non-Maximum Suppression)
        keep_idx = nms(boxes, scores, iou_threshold)
        if len(keep_idx) > 0 and (scores >= 0.89).sum() > 1:
            keep_last = scores >= 0.89
            final_box = boxes[keep_last]
            final_score = scores[keep_last]
            final_label = labels[keep_last]
        elif len(keep_idx) > 0:
            best_idx = keep_idx[0]
            final_box = boxes[best_idx]
            final_score = scores[best_idx]
            final_label = labels[best_idx]
        else:
            try:
                final_box = boxes_all[0]
                final_score = scores_all[0]
                final_label = labels_all[0]
            except Exception as e:
                print(prediction)
                print(keep_idx)
                raise e

        new_predictions.append({
            'boxes': final_box,
            'scores': final_score,
            'labels': final_label
        })

    return new_predictions

def show_image(path, model, target_size):
    global label
    image = Image.open(path).resize(target_size)
    # image = tensorflow.keras.preprocessing.image.load_img(path, target_size=target_size)
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    filtered_predictions = apply_nms(prediction, iou_threshold=0.5, score_threshold=0.3)

    boxes = filtered_predictions[0]['boxes']
    labels = filtered_predictions[0]['labels']
    scores = filtered_predictions[0]['scores']
    threshold = 0.4

    filtered_boxes = boxes[scores > threshold]
    filtered_labels = labels[scores > threshold]
    filtered_scores = scores[scores > threshold]

    draw = ImageDraw.Draw(image)
    for box, label_img, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 15), f'{class_names_step_two[label_img.item() - 1]} (prob: {score:.2f})', fill="red")

    new_size = (500, 500)
    img_resized = image.resize(new_size)
    img_tk = ImageTk.PhotoImage(img_resized)

    label.config(image=img_tk)
    label.image = img_tk


def show_result(path):
    img = Image.open(path).resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pred_prob = model_step_one.predict(img_array, verbose=0)[0][0]
    cancer = pred_prob < 0.5
    if cancer:
        show_image(path, model_step_two, target_size)

    else:
        draw = ImageDraw.Draw(img)
        # Draw the text on the image
        draw.text((0, 0 + 10), f'{class_names_step_one[cancer]} (prob: {pred_prob:.2f})', fill="red")
        new_size = (500, 500)
        img_resized = img.resize(new_size)
        img_tk = ImageTk.PhotoImage(img_resized)
        label.config(image=img_tk)
        label.image = img_tk

def open_image():
    # Open a file dialog and get the selected file path (allow PNG files)
    file_path = filedialog.askopenfilename(
        title="Open an image",
        filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
    )

    if file_path:
        try:
            show_result(file_path)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while opening the image: {e}")
    else:
        messagebox.showwarning("No file selected", "You did not select any file.")

# Create the main application window
root = tk.Tk()
root.title("Image cancer test")

# Set the window size
root.geometry("900x600")

# Create a button that opens the file picker to select an image
button = tk.Button(root, text="Open Image", command=open_image)
button.pack(pady=20)

# Create a Label widget to display the image
label = tk.Label(root)
label.pack(padx=20, pady=10, expand=True)

# Start the Tkinter event loop
root.mainloop()

