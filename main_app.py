import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageTk

selected_model = ''
epochs_entry = None
training_thread = None
training_in_progress = False

# Function to train the YOLO model
def train_model():
    global selected_model
    global epochs_entry
    global progress_bar
    global training_thread
    global training_in_progress
    global progress_window
    global current_epoch
    
    if training_in_progress:
        messagebox.showerror('Error', 'Training is already in progress')
        return
    
    # Check if model was selected
    if selected_model == '':
        messagebox.showerror('Error', 'Please select a model to train')
        return
    
    # Check if epochs entry is valid
    try:
        epochs = int(epochs_entry.get())
        if epochs <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror('Error', 'Please enter a valid number of epochs')
        return
    
    # Disabling the train button while training
    train_button.config(state=tk.DISABLED)

    print('Training model:', selected_model)
    print('Epochs:', epochs)

    training_thread = threading.Thread(target=perform_training, args=(selected_model, epochs))
    training_thread.start()

    training_in_progress = True

    window.after(100, check_training_status)

def check_training_status():
    global training_in_progress
    global window
    global training_thread

    if training_thread and training_thread.is_alive():
        window.after(500, check_training_status)
    else:
        messagebox.showinfo('Success', 'Training completed')
        training_in_progress = False
        train_button.config(state=tk.NORMAL)

def perform_training(selected_model, epochs):
    global training_in_progress
    global training_thread

    # Load the model\
    model = YOLO(selected_model+'.yaml')
    model = YOLO(selected_model+'.pt')
    model = YOLO(selected_model + '.yaml').load(selected_model + '.pt')
    
    # Train the model
    # Start the training
    model.train(data='data.yaml', epochs=epochs, imgsz=640)
    
    training_thread = None
    training_in_progress = False
    messagebox.showinfo('Success', 'Training completed')
    train_button.config(state=tk.NORMAL)

def select_file_for_plotting():
    file_path = filedialog.askopenfilename(filetypes=(("PNG Files", "*.png"), ("All Files", "*.*")))
    if file_path:
        plot_file(file_path)

def plot_parameters():
    select_file_for_plotting()

def plot_file(file_path):
    # Showing a message box to inform the user that the file was uploaded
    messagebox.showinfo("File Upload", "The file was uploaded successfully!")
    if file_path:
        # Showing the image
        plt.imshow(plt.imread(file_path))
        plt.show()

def upload_file():
    # Showing a file dialog to select the file
    file_path = filedialog.askopenfilename()
    # Showing a message box to inform the user that the file was uploaded
    messagebox.showinfo("File Upload", "The file was uploaded successfully!")
    if file_path:
        model = load_yolo_model(selected_model)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detect_objects(image, model)
        display_image_with_boxes(image, detections)

def load_yolo_model(selected_model):
    # Get all the train folders in the 'runs/detect' directory
    detect_folder = "runs/detect"
    train_folders = [folder for folder in os.listdir(detect_folder) if folder.startswith("train")]
    if not train_folders:
        messagebox.showerror("Error", "No train folders found")
        return None

    # Find the latest train folder based on modification time
    latest_train_folder = max(train_folders, key=lambda folder: os.path.getctime(os.path.join(detect_folder, folder)))

    # Path to the weights file of the model
    weights_path = os.path.join(detect_folder, latest_train_folder, "weights", "best.pt")

    # Perform Object Detection
    model = YOLO(selected_model+'.yaml')
    model = YOLO(weights_path)
    return model

def detect_objects(image, model):
    # Detecting the objects in the image
    detections = model(image)
    return detections

def display_image_with_boxes(image, detections):
    # Showing the image with the bounding boxes
    predicted = detections[0].plot()

    # Convert the image to PIL format
    predicted_pl = Image.fromarray(predicted)

    # Convert the image to Tk format
    predicted_tk = ImageTk.PhotoImage(image=predicted_pl)

    # Create a new window to display the image
    window_display = tk.Toplevel(window)
    window_display.title("Image with bounding boxes")

    # Create a label to display the image
    image_label = tk.Label(window_display, image=predicted_tk)
    image_label.pack()

    # Update the image label
    image_label.image = predicted_tk

def on_model_selected(*args):
    global selected_model
    selected_model = model_var.get()
    print('Selected model: ',selected_model)

# Create the main window
window = tk.Tk()
window.title("Object Detection Graduate Project")

# Select YOLO model and dropdown menu
model_label = tk.Label(window, text="Select YOLO model:")
model_label.pack()

model_var = tk.StringVar()
model_var.trace('w', on_model_selected)
model_dropdown = tk.OptionMenu(window, model_var, "yolov8n","yolov8s", "yolov8m", "yolov8l", "yolov8x")
model_dropdown.pack()

# Epochs label and entry
epochs_label = tk.Label(window, text="Enter number of epochs:")
epochs_label.pack()

epochs_entry = tk.Entry(window)
epochs_entry.pack()

# Train model button
train_button = tk.Button(window, text="Train Model", command=train_model)
train_button.pack() 

# Plot parameters button
plot_button = tk.Button(window, text="Plot Parameters", command=plot_parameters)
plot_button.pack()

# Upload file
upload_button = tk.Button(window, text="Upload File", command=upload_file)
upload_button.pack()

# Run the main loop
if __name__ == '__main__':
    window.mainloop()