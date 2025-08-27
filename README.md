# Virtual Glasses Try-On System

This is a real-time virtual try-on application that uses a webcam to place digital glasses onto a user's face. The system is built with Python, using **Flask** for the backend and **OpenCV** for computer vision. The frontend is a simple web interface with **HTML, CSS, and JavaScript**.

## Architecture

The application follows a **client-server architecture**.

* **Frontend (Client)**: A web browser that displays a live video feed and a set of glasses thumbnails. It handles user interactions, like selecting a different pair of glasses or uploading a new image.
* **Backend (Server)**: A Flask application that runs on a local machine. It manages the webcam stream, performs all the heavy image processing, and serves the web page.

### How it Works

The system's functionality is divided between the frontend and the backend to ensure a smooth, real-time experience.

1.  **Video Stream**: When the user opens the web page, the frontend sends a request to the `/video_feed` endpoint on the Flask server.
2.  **Frame Generation**: The backend opens the computer's webcam and starts a loop to capture frames. For each frame, it performs the following steps:
    * **Face Detection**: It uses a pre-trained **Haar Cascade classifier** to detect the face in the frame. 
    * **Image Processing**: It resizes the selected glasses image to fit the detected face and overlays the glasses onto the frame.
    * **Streaming**: The processed frame is encoded as a JPEG image and sent back to the browser in a continuous stream.
3.  **Glasses Selection**: The frontend displays a gallery of glasses. When a user clicks a thumbnail, a JavaScript function sends a request to the `/select_glasses` endpoint on the server, telling it which glasses to use for the overlay. This allows the user to change glasses in real-time without reloading the page.

## Setup Instructions 

### 1. Project Structure

Ensure your project directory is organized as follows:

```markdown
/virtual_try_on_project
|-- run_script.py
|-- /glasses
|   |-- glasses1.png
|   |-- glasses2.png
|-- /templates
|   |-- index.html
|-- /static
|   |-- /glasses
|       |-- glasses1.png
|       |-- glasses2.png

-   **`run_script.py`**: The main Python application.
-   **`/glasses`**: A folder containing the glasses images with transparent backgrounds. This is used by the backend.
-   **`/templates`**: A folder for your HTML files.
-   **`/static/glasses`**: A copy of your glasses images. Flask serves these to the frontend for the thumbnails.

### 2. Dependencies

You'll need to install the required Python libraries. Open your terminal or command prompt and run:

```bash
pip install Flask numpy opencv-python
```

### 3. Running the Application 

1. Make sure your webcam isn't being used by any other applications.
2. Navigate to your project's root directory in the terminal.
3. Run the Flask application with the following command:

```bash
python run_script.py
```
4. Once the server is running, open your browser and go to the URL provided below:

```bash
http://127.0.0.1:8000
```
5. Experience the virtual try-on
You should now see a live video feed from your webcam with the virtual glasses overlaid on your face. 
