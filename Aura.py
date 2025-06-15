import datetime
import os
import pickle
import smtplib
import sqlite3
import time
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from cryptography.fernet import Fernet

import cv2
import mediapipe as mp
import numpy as np
import psutil
import pyautogui
import pyjokes
import pyttsx3
import requests
import speech_recognition as sr
import webbrowser
from apscheduler.schedulers.background import BackgroundScheduler
from geopy.geocoders import Nominatim
from googletrans import Translator
from newsapi import NewsApiClient
from newsapi.newsapi_client import NewsApiClient
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pywhatkit as kit
import paho.mqtt.client as mqtt

# Constants
GEMINI_API_KEY = "AIzaSyBvBMlOqMvG75P7cNoSf9_GrD3_u8fw6Do"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
WEATHER_API_KEY = "8a5e3a6e8d3709dc3af233d4a3cd8d94"
WAKE_WORDS = ["hey jarvis", "hello jarvis", "wakeup jarvis", "hey", "hey jar","jarvis"]
REMINDERS_FILE = "reminders.pkl"
NEWS_API_KEY = "f2d42491b0354d12bb2bf28efdd536f4"  # Replace with your actual key
MQTT_BROKER = "your_mqtt_broker"  # Replace with your MQTT broker address

# Initialize components
engine = pyttsx3.init()
recognizer = sr.Recognizer()
geolocator = Nominatim(user_agent="jarvis_assistant")
scheduler = BackgroundScheduler()
scheduler.start()
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
translator = Translator()

# Contacts databases
CONTACTS = {
    "mom": "+918309304686",
    "dad": "+918897878447",
    "work": "boss@company.com",
    "friend": "friend@gmail.com"
}

# Language mapping for translation
LANG_MAP = {
    'hindi': 'hi',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'english': 'en'
}

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listen to microphone input and convert to text"""
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            return query.lower()
        except sr.UnknownValueError:
            print("Sorry, I did not understand.")
        except sr.RequestError:
            print("Request Error.")
        return ""

def greet():
    """Greet based on time of day"""
    hour = datetime.datetime.now().hour
    if 0 <= hour < 12:
        speak("Good Morning!, SIR")
    elif 12 <= hour < 18:
        speak("Good Afternoon!, SIR")
    else:
        speak("Good Evening!, SIR")

def system_report():
    """Provide system status report"""
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    battery = psutil.sensors_battery()
    battery_status = f"{battery.percent}%" if battery else "N/A"
    speak(f"CPU usage is at {cpu} percent. Memory usage is {memory} percent. Battery is at {battery_status}.")

def get_weather(city="Hyderabad"):
    """Fetch weather information for a city"""
    if not WEATHER_API_KEY:
        return "API key for weather is not set."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url)
        data = res.json()
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"The temperature in {city} is {temp}°C with {desc}."
    except Exception as e:
        print(f"Weather API error: {e}")
        return "I couldn't fetch the weather."

def smart_reply(prompt):
    """Generate AI response using Gemini API"""
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {"contents": [{"parts": [{"text": prompt + " in 50 words"}]}]}
    try:
        res = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data)
        res.raise_for_status()
        response = res.json()
        if "candidates" in response and response["candidates"]:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        return "I'm not sure how to respond to that."
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error contacting Gemini AI."

def send_whatsapp_message():
    """Send WhatsApp message to a contact by name"""
    try:
        speak("Who should I message?")
        contact_name = listen().lower().strip()

        # Check if the contact exists
        phone_number = CONTACTS.get(contact_name)
        if not phone_number:
            speak(f"I couldn't find a contact named {contact_name}. Please try again.")
            return
        
        speak(f"What should I say to {contact_name}?")
        message = listen()

        if not message:
            speak("No message detected.")
            return

        speak(f"Sending message to {contact_name}: {message}")
        kit.sendwhatmsg_instantly(phone_number, message, wait_time=15)
        pyautogui.press('enter')
        speak("Message sent successfully!")

    except Exception as e:
        print(f"WhatsApp error: {e}")
        speak("Failed to send WhatsApp message.")

def send_email():
    """Send email using voice commands"""    
    try:
        speak("Who should I email?")
        recipient_name = listen().lower()
        recipient = CONTACTS.get(recipient_name)
        
        if not recipient:
            speak(f"No contact found for {recipient_name}")
            return
            
        speak("What should the subject be?")
        subject = listen()
        
        speak("What should I say in the email?")
        body = listen()
        
        # Configure your email settings
        msg = MIMEMultipart()
        msg['From'] = "your_email@gmail.com"  # Replace with your email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send the email (configure your SMTP settings)
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login("your_email@gmail.com", "your_password")  # Replace with your credentials
            server.send_message(msg)
            
        speak("Email sent successfully!")
        
    except Exception as e:
        print(f"Email error: {e}")
        speak("Failed to send email")

def read_news():
    """Read news headlines"""
    try:
        speak("Which category? Business, technology, sports, or general?")
        category = listen().lower()
        
        valid_categories = ['business', 'technology', 'sports', 'general']
        if category not in valid_categories:
            category = 'general'
            
        top_headlines = newsapi.get_top_headlines(category=category,
                                                language='en',
                                                country='in')
        
        for i, article in enumerate(top_headlines['articles'][:5]):
            speak(f"News {i+1}: {article['title']}")
            time.sleep(1)
            
    except Exception as e:
        print(f"News error: {e}")
        speak("Failed to fetch news")

def set_reminder():
    """Set a voice reminder"""
    try:
        speak("What should I remind you about?")
        reminder_text = listen()
        
        speak("When should I remind you? Say something like 'in 30 minutes'")
        reminder_time = listen()  # Would need natural language processing
        
        # Load existing reminders
        reminders = []
        if Path(REMINDERS_FILE).exists():
            with open(REMINDERS_FILE, 'rb') as f:
                reminders = pickle.load(f)
                
        # Add new reminder
        reminders.append({
            'text': reminder_text,
            'time': reminder_time  # Would need proper datetime conversion
        })
        
        # Save reminders
        with open(REMINDERS_FILE, 'wb') as f:
            pickle.dump(reminders, f)
            
        speak(f"Reminder set: {reminder_text}")
        
    except Exception as e:
        print(f"Reminder error: {e}")
        speak("Failed to set reminder")

def home_automation():
    """Control smart home devices"""
    try:
        speak("Which device? Lights, fan, or AC?")
        device = listen().lower()
        
        speak(f"What action for {device}? On or off?")
        action = listen().lower()
        
        # MQTT setup for IoT devices
        client = mqtt.Client()
        client.connect(MQTT_BROKER, 1883, 60)
        
        topic = f"home/{device}"
        client.publish(topic, action)
        
        speak(f"Turning {device} {action}")
        
    except Exception as e:
        print(f"Home automation error: {e}")
        speak("Failed to control device")

def translate_text():
    """Translate between languages"""    
    try:
        speak("What language to translate to?")
        dest_lang = listen().lower()
        
        if dest_lang not in LANG_MAP:
            speak("Language not supported")
            return
            
        speak("What should I translate?")
        text = listen()
        
        translation = translator.translate(text, dest=LANG_MAP[dest_lang])
        speak(f"Translation: {translation.text}")
        
    except Exception as e:
        print(f"Translation error: {e}")
        speak("Failed to translate")

def track_expense():
    """Track expenses via voice"""
    try:
        speak("What's the amount?")
        amount = listen()
        
        speak("What category? Food, transport, shopping?")
        category = listen().lower()
        
        speak("Any additional notes?")
        notes = listen()
        
        # Store in SQLite database
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS expenses
                     (date text, amount real, category text, notes text)''')
        
        c.execute("INSERT INTO expenses VALUES (date('now'), ?, ?, ?)",
                 (float(amount), category, notes))
        conn.commit()
        conn.close()
        
        speak(f"Added {amount} to {category} expenses")
        
    except Exception as e:
        print(f"Expense tracking error: {e}")
        speak("Failed to track expense")

def health_tips():
    """Provide health tips based on time of day"""
    hour = datetime.datetime.now().hour
    
    if 6 <= hour < 12:
        speak("Good morning! Remember to drink water and stretch.")
    elif 12 <= hour < 14:
        speak("Time for lunch! Choose something healthy.")
    elif 14 <= hour < 18:
        speak("Consider taking a short walk to stay active.")
    else:
        speak("Time to wind down. Avoid screens before bed.")

def password_manager():
    """Securely store and retrieve passwords"""
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    
    try:
        speak("Add or get password?")
        action = listen().lower()
        
        if 'add' in action:
            speak("For which service?")
            service = listen()
            
            speak("What's the username?")
            username = listen()
            
            speak("What's the password?")
            password = listen()
            
            # Encrypt and store
            encrypted = cipher_suite.encrypt(password.encode())
            # Store in secure database
            
            speak(f"Credentials for {service} saved securely")
            
        elif 'get' in action:
            speak("For which service?")
            service = listen()
            
            # Retrieve from database and decrypt
            # decrypted = cipher_suite.decrypt(encrypted).decode()
            
            speak(f"Credentials for {service} are...")  # Would read actual credentials
            
    except Exception as e:
        print(f"Password manager error: {e}")
        speak("Password operation failed")

def screen_recording():
    """Record screen with voice commands"""
    try:
        speak("Starting screen recording. Say 'stop recording' to end.")
        
        screen_size = pyautogui.size()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("output.avi", fourcc, 20.0, screen_size)
        
        recording = True
        while recording:
            img = pyautogui.screenshot()
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
            
            # Check for stop command
            with sr.Microphone() as source:
                audio = recognizer.listen(source, phrase_time_limit=1)
                try:
                    command = recognizer.recognize_google(audio).lower()
                    if "stop recording" in command:
                        recording = False
                except:
                    pass
                    
        out.release()
        speak("Screen recording saved")
        
    except Exception as e:
        print(f"Recording error: {e}")
        speak("Failed to record screen")

def object_detection():
    """Optimized object detection with age and gender recognition"""
    try:
        # Model file paths
        model_files = {
            'yolo': ('yolov3.weights', 'yolov3.cfg', 'coco.names'),
            'age': ('age_deploy.prototxt', 'age_net.caffemodel'),
            'gender': ('gender_deploy.prototxt', 'gender_net.caffemodel')
        }
        
        # Verify all required files exist
        missing_files = []
        for model_type, files in model_files.items():
            for file in files:
                if not os.path.exists(file):
                    missing_files.append(file)
        
        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}"
            print(error_msg)
            speak("Error: Missing model files for object detection")
            return

        # Load YOLO model
        print("Loading YOLO model...")
        net = cv2.dnn.readNet(model_files['yolo'][0], model_files['yolo'][1])
        
        # Check if CUDA is available
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using GPU acceleration")
            else:
                print("Using CPU (no CUDA devices found)")
        except:
            print("Using CPU (CUDA not available)")
        
        # Load class names
        with open(model_files['yolo'][2], 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Load age and gender models
        print("Loading age and gender models...")
        age_net = cv2.dnn.readNetFromCaffe(model_files['age'][0], model_files['age'][1])
        gender_net = cv2.dnn.readNetFromCaffe(model_files['gender'][0], model_files['gender'][1])

        # Labels
        AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                     '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        GENDER_LIST = ['Male', 'Female']
        
        # Detection parameters
        conf_threshold = 0.5  # Minimum confidence for detection
        nms_threshold = 0.4   # Non-maximum suppression threshold
        
        # Get output layer names (handles different OpenCV versions)
        layer_names = net.getLayerNames()
        try:
            # OpenCV 4.x
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        except:
            # OpenCV 3.x
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        def detect_age_gender(face_img, frame, x, y):
            """Detect age and gender from face region"""
            try:
                if face_img.size == 0:
                    return
                    
                # Create blob from face image
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                            (78.4, 87.8, 114.9), swapRB=False)
                
                # Gender detection
                gender_net.setInput(blob)
                gender_pred = gender_net.forward()
                gender = GENDER_LIST[gender_pred[0].argmax()]
                
                # Age detection
                age_net.setInput(blob)
                age_pred = age_net.forward()
                age = AGE_RANGES[age_pred[0].argmax()]

                # Display results
                cv2.putText(frame, f"{gender}, {age}", 
                            (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)
            except Exception as e:
                print(f"Age/gender detection error: {str(e)}")

        # Initialize video capture
        print("Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            speak("I cannot access the camera")
            return
            
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # YOLO input size (416x416 for YOLOv3-416)
        yolo_input_size = (416, 416)
        
        print("Starting detection loop...")
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, yolo_input_size, 
                                       swapRB=True, crop=False)
            net.setInput(blob)
            
            # Run detection
            start_time = time.time()
            try:
                outputs = net.forward(output_layers)
            except Exception as e:
                print(f"Detection error: {str(e)}")
                break
                
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            
            # Process detections
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > conf_threshold:
                        # Scale box coordinates to original image
                        box = detection[0:4] * np.array([width, height, width, height])
                        (center_x, center_y, box_width, box_height) = box.astype("int")
                        
                        x = int(center_x - (box_width / 2))
                        y = int(center_y - (box_height / 2))
                        
                        boxes.append([x, y, int(box_width), int(box_height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            try:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            except Exception as e:
                print(f"NMS error: {str(e)}")
                indices = []
            
            # Draw detections
            for i in indices:
                try:
                    # Handle both OpenCV 4.x and 3.x NMS returns
                    idx = i if isinstance(i, (int, np.integer)) else i[0]
                    
                    x, y, w, h = boxes[idx]
                    label = f"{classes[class_ids[idx]]} {confidences[idx]:.1f}"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Detect age/gender for people
                    if classes[class_ids[idx]] == "person":
                        face_img = frame[y:y+h, x:x+w]
                        if face_img.size > 0:
                            detect_age_gender(face_img, frame, x, y)
                except Exception as e:
                    print(f"Drawing error: {str(e)}")
                    continue
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("Object Detection", frame)
            
            # Exit on ESC or 'q'
            key = cv2.waitKey(1)
            if key in [27, ord('q')]:
                break

    except Exception as e:
        print(f"Error in object detection: {str(e)}")
        speak("Sorry, I encountered an error in object detection")
        
    finally:
        # Clean up
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Object detection stopped")

def video_control():
    """Control mouse with hand gestures"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()

    try:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get finger positions
                    landmarks = {
                        'index': hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        'thumb': hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                        'middle': hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                        'ring': hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                        'pinky': hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    }
                    
                    # Convert to pixel coordinates
                    coords = {k: (int(lm.x * w), int(lm.y * h)) for k, lm in landmarks.items()}
                    
                    # Move mouse to index finger
                    mouse_x, mouse_y = int(landmarks['index'].x * screen_width), int(landmarks['index'].y * screen_height)
                    pyautogui.moveTo(mouse_x, mouse_y)
                    
                    # Calculate distances for gestures
                    dist_thumb_index = np.hypot(coords['thumb'][0]-coords['index'][0], 
                                              coords['thumb'][1]-coords['index'][1])
                    dist_index_middle = np.hypot(coords['index'][0]-coords['middle'][0], 
                                               coords['index'][1]-coords['middle'][1])
                    dist_thumb_ring = np.hypot(coords['thumb'][0]-coords['ring'][0], 
                                             coords['thumb'][1]-coords['ring'][1])
                    dist_thumb_pinky = np.hypot(coords['thumb'][0]-coords['pinky'][0], 
                                              coords['thumb'][1]-coords['pinky'][1])

                    # Handle gestures
                    if dist_thumb_index < 15:
                        pyautogui.click()
                    elif dist_index_middle < 15:
                        pyautogui.rightClick()
                    elif dist_thumb_ring < 15:
                        pyautogui.scroll(30)
                        cv2.putText(frame, "Scroll Up", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif dist_thumb_pinky < 15:
                        pyautogui.scroll(-30)
                        cv2.putText(frame, "Scroll Down", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, "Press 'Q' or 'Esc' to Exit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Virtual Mouse Control", frame)

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Video control error: {e}")
        speak("Error in video control")

def skip_youtube_ads():
    """Skip YouTube ads automatically"""
    try:
        time.sleep(2)
        skip_button = pyautogui.locateOnScreen('skip_ad_button.png', confidence=0.7)
        if skip_button:
            pyautogui.click(skip_button)
            speak("Skipped an ad!")
            time.sleep(2)
    except Exception as e:
        print(f"Ad skip error: {e}")

def get_location_info(place_name):
    """Get location information and show on map"""
    try:
        location = geolocator.geocode(place_name, timeout=10)
        if location:
            lat, lon = location.latitude, location.longitude
            address = location.address
            speak(f"Found {place_name} at {address}. Latitude {lat:.4f}, Longitude {lon:.4f}.")
            webbrowser.open(f"https://www.google.com/maps/search/?api=1&query={lat},{lon}")
        else:
            speak(f"Sorry, I couldn't find the location for '{place_name}'.")
    except Exception as e:
        print(f"Location error: {e}")
        speak("Sorry, I encountered an error while looking up the location.")

def handle_command(command):
    """Process user commands"""
    if not command:
        return "continue"
        
    command = command.lower()
    
    # Basic responses
    if "your name" in command:
        speak("I am JARVIS, your artificial intelligence assistant.")
    elif "time" in command:
        speak("The time is " + time.strftime("%I:%M %p"))

    elif "weather" in command:
        city = "Hyderabad"
        if "weather in" in command:
            city = command.split("weather in")[-1].strip()
        elif "weather for" in command:
            city = command.split("weather for")[-1].strip()
        speak(get_weather(city))
    elif command.startswith("where is"):
        place = command.replace("where is", "").strip()
        if place:
            get_location_info(place)
        else:
            speak("Where is what? Please specify a place.")
    elif "how are you" in command:
        speak("Functioning at full capacity, sir.")
    elif "thank you" in command:
        speak("At your service, always.")
    elif "system report" in command:
        system_report()
    elif "shutdown" in command:
        speak("Scheduling shutdown in five minutes.")
        os.system("shutdown -s -t 300")
    elif "cancel shutdown" in command:
        speak("Cancelling scheduled shutdown.")
        os.system("shutdown -a")
    elif "exit" in command or "quit" in command:
        speak("Shutting down JARVIS interface.")
        return "exit"
        
    # Application control
    elif "open notepad" in command:
        os.system("notepad.exe")
    elif "open chrome" in command:
        os.system("start chrome")
    elif "open command prompt" in command:
        os.system("start cmd")
    elif "close chrome" in command:
        os.system("taskkill /f /im chrome.exe")
        speak("Chrome closed.")
    elif "close youtube" in command or "close browser" in command:
        os.system("taskkill /f /im chrome.exe")
        os.system("taskkill /f /im msedge.exe")
        speak("YouTube or browser closed.")
    elif "close notepad" in command:
        os.system("taskkill /f /im notepad.exe")
        speak("Notepad closed.")
        
    # Web operations
    elif "open google" in command:
        webbrowser.open("https://www.google.com")
    elif "search google for" in command:
        query = command.replace("search google for", "").strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        speak(f"Searching Google for {query}")
    elif "play music" in command or "play song" in command:
        speak("Playing your playlist.")
        webbrowser.open("https://www.youtube.com/watch?v=vzP7vtGqqeA&list=RDvzP7vtGqqeA&start_radio=1")
    elif "open youtube" in command:
        speak("What should I search on YouTube?")
        query = listen()
        if query:
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            speak(f"Searching YouTube for {query}")
        else:
            webbrowser.open("https://www.youtube.com")
            speak("Opening YouTube homepage")
            
    # Media control
    elif "play first video" in command:
        pyautogui.moveTo(300, 400)
        pyautogui.click()
    elif "play second video" in command:
        pyautogui.moveTo(300, 500)
        pyautogui.click()
    elif "play third video" in command:
        pyautogui.moveTo(300, 600)
        pyautogui.click()
    elif "pause" in command or "play video" in command:
        pyautogui.press("space")
    elif "volume up" in command:
        pyautogui.press("volumeup")
    elif "volume down" in command:
        pyautogui.press("volumedown")
    elif "mute" in command:
        pyautogui.press("volumemute")
    elif "next" in command:
        pyautogui.press("nexttrack")
    elif "previous" in command:
        pyautogui.press("prevtrack")
    elif "brightness up" in command:
        pyautogui.press("brightnessup")
    elif "brightness down" in command:
        pyautogui.press("brightnessdown")
        
    # Special features
    elif "type" in command:
        speak("What should I type?")
        pyautogui.write(listen())
    elif "mouse" in command:
        speak("Giving access to mouse!")
        video_control()
    elif "see" in command:
        speak("Activating object detection!")
        object_detection()
    elif "skip ad" in command:
        skip_youtube_ads()
    elif "whatsapp" in command or "message" in command:
        send_whatsapp_message()
    elif "email" in command:
        send_email()
    elif "news" in command:
        read_news()
    elif "remind" in command:
        set_reminder()
    elif "home" in command and ("control" in command or "automation" in command):
        home_automation()
    elif "translate" in command:
        translate_text()
    elif "expense" in command:
        track_expense()
    elif "health" in command or "tip" in command:
        health_tips()
    elif "password" in command:
        password_manager()
    elif "record" in command and "screen" in command:
        screen_recording()
    else:
        speak(smart_reply(command))
        
    return "continue"

def active_mode():
    """Handle active listening mode"""
    speak("I'm listening, SIR")
    active_start_time = time.time()
    silence_start_time = None
    ACTIVE_TIMEOUT = 180  # 3 minutes
    SILENCE_TIMEOUT = 30  # 30 seconds

    while True:
        current_time = time.time()
        
        # Check timeouts
        if (current_time - active_start_time) > ACTIVE_TIMEOUT:
            speak("It's been quiet for a while. Going back to sleep.")
            break
        if silence_start_time and (current_time - silence_start_time) > SILENCE_TIMEOUT:
            speak("Going silent now. Say my name to wake me.")
            break
            
        # Listen for commands
        command = listen()
        if command:
            silence_start_time = current_time
            if handle_command(command) == "exit":
                return "exit"
        else:
            if not silence_start_time:
                silence_start_time = current_time
                
        time.sleep(0.1)
        
    return "continue"

def main():
    """Main program loop"""
    speak("Initializing JARVIS systems.")
    time.sleep(1)
    speak("Say 'Hey Jarvis' to wake me up.")
    print("Listening for wake word...")
    
    while True:
        command = listen()
        print("Heard:", command)
        
        if any(wake in command for wake in WAKE_WORDS):
            greet()
            if active_mode() == "exit":
                break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        speak("Shutting down JARVIS.")
    except Exception as e:
        print(f"Fatal error: {e}")
        speak("Critical error occurred. Restarting systems.")
        main()