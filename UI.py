import streamlit as st
from streamlit_option_menu import option_menu
import os
import subprocess
import time
import sys
import json

# Display a loading bar before showing the menu
progress_text = '<div class="loading-text"> </div>'
st.markdown(progress_text, unsafe_allow_html=True)

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.01)  # Simulates loading time
    my_bar.progress(percent_complete + 1)
time.sleep(1)
my_bar.empty()

with st.sidebar:
    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR9XwMdtD9VIelkI8H2AroAsqXGTL57ZSavKA&s", 
        use_container_width=True, 
        caption="The rise of Deepfakes"
    )
    st.markdown(
        """
        <div style="font-size:18px; font-family:Arial, sans-serif; color:rgb(255,255,255); text-align:justify;">
        <b>Deepfakes</b>, manipulated media using AI, pose significant threats to society by spreading misinformation and undermining trust.
        <br><br>
        <b>Deepfake detection</b> is crucial to combat these dangers. Techniques involve analyzing subtle artifacts like inconsistencies in blinking, 
        breathing patterns, and inconsistencies in lighting and shadows. 
        <br><br>
        Machine learning models are trained on vast datasets of real and manipulated videos to identify these anomalies, helping to distinguish 
        genuine content from fabricated ones.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
    <style>
    .progress-text {
        font-size: 20px; /* Larger text size for progress */
        font-weight: bold;
        color: #444444;
        text-align: center;
        margin-top: 20px;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
        padding: 0;
    }
    .title {
        font-size: 60px;
        font-weight: bold;
        font-family: Monospace;
        color: #8cf30f;
    }
    .header {
        font-size: 40px;
        font-weight: bold;
        font-family: 'Verdana', sans-serif;
        color: #8cf30f;
        margin-bottom: 20px;
    }
    .footer {
        font-size: 14px;
        text-align: left;
        margin-top: 100px;
        color: #888888;
    }
    .sidebar {
        font-size: 25px;
        font-family: 'Arial', sans-serif;
    }
    .write {
        font-size: 45px;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add background image CSS
background_image = """
<style>
    body {
        background-image: url('https://www.cybereason.com/hubfs/Deepfakes-Blog.jpg');
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 15px;
        padding: 20px;
    }
</style>
"""

st.markdown("# DEEPFAKE Detection")
st.markdown(background_image, unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size:17px; font-family:Monospace; color:#FFFFFF;">
    Select whether you want to predict if an <b>image</b>, <b>video</b>, or <b>audio</b> is a deepfake.
    </div>
    """, 
    unsafe_allow_html=True
)
st.markdown("")

# Add the option menu on the main page
selected_option = option_menu(
    menu_title="Choose an Option",
    options=["Video", "Audio", "Image"],
    icons=["camera-video", "mic", "image"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"  # Place it horizontally on the page
)

# Display a dynamic subheading
st.subheader(f"🔍 {selected_option}")

video_command = [
    "conda activate Df &&cd /Users/anand/Desktop/ai/deepfake-detection/CViT && python cvit_prediction.py --p sample__prediction_data --f 15 --n cvit2 --fp16 y"
]

audio_command = [
    "conda activate Df && python eval.py --input_path /Users/anand/Desktop/ai/deepfake-detection/audio/Synthetic-Voice-Detection-Vocoder-Artifacts-main/male.wav --model_path /Users/anand/Desktop/ai/deepfake-detection/audio/Synthetic-Voice-Detection-Vocoder-Artifacts-main/lambda0.5_epoch_25.pth"
]

image_command = [
    "conda activate Df && python /path/to/your/image_detection_script.py --image_path /path/to/your/uploaded_image.jpg"
]

def run_command_in_new_terminal(command):
    if os.name == 'nt':  # Windows
        # Start a new cmd instance for the command
        subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', command], shell=True)
    elif os.name == 'posix' and 'darwin' in sys.platform:  # macOS
        # Use AppleScript to open a new terminal and run the command
        apple_script = f"""
        tell application "Terminal"
            do script "{command}"
            activate
        end tell
        """
        subprocess.Popen(["osascript", "-e", apple_script])
    elif os.name == 'posix' and 'linux' in sys.platform:  # Linux
        # Use xterm or gnome-terminal
        terminal_command = ['xterm', '-hold', '-e', command]
        try:
            subprocess.Popen(terminal_command)
        except FileNotFoundError:
            print("xterm is not installed. Please install it or use another terminal emulator.")
    else:
        print("Unsupported OS for terminal execution.")


# Function to access and display the JSON file
def display_prediction_json(json_file_path):
    try:
        # Check if the JSON file exists
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                data = json.load(f)  # Load the JSON content
            
            # Display the JSON content on the Streamlit app
            st.markdown("### Prediction Results")
            st.json(data)  # Display JSON data in a nicely formatted way
        else:
            st.error(f"JSON file not found at: {json_file_path}")
    except Exception as e:
        st.error(f"Failed to read the JSON file: {e}")

# Display model details based on the selected option
if selected_option == "Video":
    st.markdown('<div class="header">Deepfake detection of Video</div>', unsafe_allow_html=True)
    st.markdown("## Model : CViT")
    st.markdown("""
        <div style="font-size:20px; font-family:Monospace; color:#FFFFFF;">
         Here's a concise description of the layers in the CViT (Convolutional Vision Transformer) architecture:

        - <u>**Features Layer:**</u> A deep convolutional network with 5 stages of convolution blocks.
        - <u>**Patch Embedding Layer:**</u> Transforms image patches into linear embeddings.
        - <u>**Positional Embedding:**</u> Adds learnable positional embeddings for spatial relationships.
        - <u>**Transformer Layer:**</u> Implements self-attention mechanisms with residual connections.
        - <u>**Classification Head:**</u> Produces final class predictions through linear layers.
        
        The CViT network combines convolutional feature extraction with transformer-style sequence processing.
        </div>
    """, unsafe_allow_html=True)

    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file:
        st.write("Processing the uploaded video...")
        
        # Ensure the directory exists
        output_dir = "sample__prediction_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the video file
        video_path = os.path.join(output_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Verify that the file exists
        if os.path.exists(video_path):
            st.success(f"Video saved at: {video_path}")
            
            # Run the deepfake detection command in a new terminal
            command = [
                "cd /Users/anand/Desktop/ai/deepfake-detection/CViT",
                "conda activate Df",
                "python cvit_prediction.py --p sample__prediction_data --f 15 --n cvit2 --fp16 y"
            ]
            try:
                run_command_in_new_terminal(" && ".join(command))
                st.success("Command executed in a new terminal. Check the output there.")
            except Exception as e:
                st.error(f"Failed to execute command: {e}")
        else:
            st.error("Failed to save the video. Please try again.")
    
    # Path to the JSON file containing predictions
    json_file_path = "/Users/anand/Desktop/ai/deepfake-detection/CViT/result/prediction.json"

    # Button to display prediction JSON
    if st.button("Show Prediction Results"):
        display_prediction_json(json_file_path)



elif selected_option == "Audio":
    st.markdown('<div class="header">Deepfake detection of Audio</div>', unsafe_allow_html=True)
    st.markdown("## Model : RawNet")
    st.markdown("""
        <div style="font-size:20px; font-family:Monospace; color:#FFFFFF;">
        Here's a concise description of the layers in the RawNet architecture:
        
        - <u>**SincConv Layer:**</u> Creates learnable bandpass filters using the Mel scale.
        - <u>**Residual Blocks:**</u> Processes and transforms the feature representations.
        - <u>**Attention Mechanism:**</u> Dynamically weights feature importance with sigmoid scaling.
        - <u>**GRU Layer:**</u> Captures temporal dependencies and extracts final representation.
        - <u>**Fully Connected Layers:**</u> Produces output probabilities for classification.</div>
    """, unsafe_allow_html=True)

    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    if audio_file:
        st.write("Processing the uploaded audio...")
        
        # Save audio locally
        audio_path = "uploaded_audio." + audio_file.name.split('.')[-1]
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Execute the command and capture terminal output
        command = [
            "cd /Users/anand/Desktop/ai/deepfake-detection/audio",
            "conda activate Df",
            "python eval.py --input_path /Users/anand/Desktop/ai/deepfake-detection/audio/Data/Boris.wav --model_path /Users/anand/Desktop/ai/deepfake-detection/audio/lambda0.5_epoch_25.pth"
        ]
        
        try:
            run_command_in_new_terminal(" && ".join(command))
            st.success("Command executed in a new terminal. Check the output there.")
        except Exception as e:
            st.error(f"Failed to execute command: {e}")
    
    # Path to the JSON file containing predictions
    json_file_path = "/Users/anand/Desktop/ai/deepfake-detection/audio/result/prediction.json"

    # Button to display prediction JSON
    if st.button("Show Prediction Results"):
        try:
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    data = json.load(f)  # Load the JSON content
                
                # Display the JSON content
                st.markdown("### Prediction Results")
                st.json(data)  # Streamlit's built-in method to display JSON
            else:
                st.error(f"JSON file not found at: {json_file_path}")
        except Exception as e:
            st.error(f"Failed to read the JSON file: {e}")


elif selected_option == "Image":
    st.markdown('<div class="header">Deepfake detection of Image</div>', unsafe_allow_html=True)
    st.markdown("## Model : Meso4")
    st.markdown("""
        <div style="font-size:20px; font-family:Monospace; color:#FFFFFF;">
        Here's a description of each layer in the Meso4 model:
        
        - <u>**Input Layer:**</u> Accepts image inputs.
        - <u>**Convolutional Layers (x1-x4):**</u> Extracts and down-samples features progressively.
        - <u>**Flatten Layer:**</u> Converts 2D feature maps to a 1D vector.
        - <u>**Dropout Layers:**</u> Prevent overfitting by randomly dropping neurons.
        - <u>**Dense Layer:**</u> Captures high-level features.
        - <u>**Output Layer:**</u> Produces a probability for binary classification.
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.write("Processing the uploaded image...")
        # Save image locally
        image_path = "uploaded_image." + uploaded_file.name.split('.')[-1]
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Placeholder command for image processing (replace with your script)
        command = [
            "cd /Users/anand/Desktop/ai/deepfake-detection/Image-deepfake",
            "conda activate Df",
            "python /Users/anand/Desktop/ai/deepfake-detection/Image-deepfake/image_detect.py"
        ]

        try:
            run_command_in_new_terminal(" && ".join(command))
            st.success("Command executed in a new terminal. Check the output there.")
        except Exception as e:
            st.error(f"Failed to execute command: {e}")
    
    # Path to the JSON file containing predictions
    json_file_path = "/Users/anand/Desktop/ai/deepfake-detection/Image-deepfake/result/prediction.json"

    # Button to display prediction JSON
    if st.button("Show Prediction Results"):
        try:
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    data = json.load(f)  # Load the JSON content
                
                # Display the JSON content
                st.markdown("### Prediction Results")
                st.json(data)  # Streamlit's built-in method to display JSON
            else:
                st.error(f"JSON file not found at: {json_file_path}")
        except Exception as e:
            st.error(f"Failed to read the JSON file: {e}")
