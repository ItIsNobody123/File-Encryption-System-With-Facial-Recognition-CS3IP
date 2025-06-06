import cv2 # Imports OpenCV for image processing
import numpy as np #imports NumPy for numerical operations
import os #imports OS for file and directory operations
import face_recognition # Imports face_recognition for face detection and recognition
from cryptography.fernet import Fernet # Imports Fernet through cyptography for file encryption
import sqlite3 # Imports SQLite for database management
import time # Imports time for time-related functions


class DatabaseManager:
    def __init__(self, db_path='face_database.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            registration_time TIMESTAMP
        )
        ''')
        self.conn.commit()
    
    def add_face(self, name, image_path):
        timestamp = int(time.time())
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO faces (name, image_path, registration_time) VALUES (?, ?, ?)",
            (name, image_path, timestamp)
        )
        self.conn.commit()
        return True
    
    def get_all_faces(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, image_path FROM faces")
        return cursor.fetchall()
    
    def close(self):
        if self.conn:
            self.conn.close()

class FaceDetector:
    def __init__(self):
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def preprocess_image(self, frame):
        """
        Preprocess the image to improve face detection in challenging conditions.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        return gray

    def detect_faces(self, frame):
        """
        Detect faces in the given frame after preprocessing.
        """
        # Preprocess the frame
        preprocessed_frame = self.preprocess_image(frame)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            preprocessed_frame,
            scaleFactor=1.1,
            minNeighbors=6,  # Adjusted for better accuracy
            minSize=(30, 30)
        )
        return faces

class FaceRecogniser:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.encoding_cache = {}  # Cache for face encodings to avoid recomputing
    
    def load_known_faces(self, face_data):
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        
        for name, image_path in face_data:
            if os.path.exists(image_path):
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        print(f"Loaded face: {name}")
                    else:
                        print(f"Warning: No faces found in {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            else:
                print(f"Warning: Image file not found: {image_path}")
    
    def recognize_faces(self, frame):
        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations using face_recognition (hog model is lighter than cnn)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        # If no faces found, return empty results
        if not face_locations:
            return []
        
        # Only compute encodings if faces are found
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognition_results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_recognized = False
            face_name = "Unknown"
            confidence = 0
            
            # Check if the face matches any known faces
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                matches = list(face_distances <= 0.6)  # Tolerance threshold
                
                if True in matches:
                    match_index = matches.index(True)
                    face_name = self.known_face_names[match_index]
                    face_recognized = True
                    # Calculate confidence percentage
                    confidence = (1 - face_distances[match_index]) * 100
            
            recognition_results.append({
                'location': (top, right, bottom, left),
                'name': face_name,
                'recognized': face_recognized,
                'confidence': confidence
            })
        
        return recognition_results

class FileEncryptor:
    def __init__(self, test_files_dir="test_files"):
        self.test_files_dir = test_files_dir
        self.key_file = "encryption_key.key"
        
        if not os.path.exists(self.test_files_dir):
            os.makedirs(self.test_files_dir)
    
    def get_cipher(self):
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as file:
                return Fernet(file.read())
        return None
    
    def encrypt_files(self):
        cipher = self.get_cipher()
        if not cipher:
            return
        
        all_files = os.listdir(self.test_files_dir)
        unencrypted_files = [f for f in all_files if not f.endswith('.encrypted')]
        
        if not unencrypted_files:
            return
            
        for filename in unencrypted_files:
            file_path = os.path.join(self.test_files_dir, filename)
            
            if os.path.isdir(file_path):
                continue
                
            try:
                with open(file_path, 'rb') as file:
                    file_data = file.read()
                
                encrypted_data = cipher.encrypt(file_data)
                
                encrypted_file_path = file_path + '.encrypted'
                with open(encrypted_file_path, 'wb') as file:
                    file.write(encrypted_data)
                
                os.remove(file_path)
                
            except Exception as e:
                print(f"Error encrypting {filename}: {e}")
        
        print("Files encrypted")
    
    def decrypt_files(self):
        cipher = self.get_cipher()
        if not cipher:
            return
        
        encrypted_files = False
        for filename in os.listdir(self.test_files_dir):
            if not filename.endswith('.encrypted'):
                continue
                
            encrypted_files = True
            encrypted_file_path = os.path.join(self.test_files_dir, filename)
            original_file_path = encrypted_file_path[:-10]
            
            try:
                with open(encrypted_file_path, 'rb') as file:
                    encrypted_data = file.read()
                
                decrypted_data = cipher.decrypt(encrypted_data)
                
                with open(original_file_path, 'wb') as file:
                    file.write(decrypted_data)
                
                os.remove(encrypted_file_path)
                
            except Exception as e:
                print(f"Error decrypting {filename}: {e}")
        
        if encrypted_files:
            print("Files decrypted")

class UserInterface:
    def __init__(self):
        self.admin_passcode = "1234"  # Change this to your secure passcode
    
    def display_menu(self):
        print("\n===== Facial Recognition Security System =====")
        print("1. Register a new face (requires admin passcode)")
        print("2. Run facial recognition in normal mode")
        print("3. Run facial recognition in background mode")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        return choice
    
    def get_admin_passcode(self):
        passcode = input("Enter admin passcode: ")
        return passcode
    
    def get_face_name(self):
        name = input("Enter name for the face: ")
        return name
    
    def draw_recognition_results(self, frame, recognition_results):
        display_frame = frame.copy()
        
        for result in recognition_results:
            top, right, bottom, left = result['location']
            face_recognized = result['recognized']
            face_name = result['name']
            confidence = result['confidence']
            
            # Draw rectangle
            color = (0, 255, 0) if face_recognized else (0, 0, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            # Display name and confidence
            if face_recognized:
                confidence_text = f"{face_name} ({confidence:.1f}%)"
            else:
                confidence_text = "Unknown"
            
            # Draw text background
            cv2.rectangle(display_frame, (left, top - 30), (right, top), color, cv2.FILLED)
            cv2.putText(display_frame, confidence_text, (left + 6, top - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display_frame

class FacialRecognitionSystem:
    def __init__(self):
        # Initialize components
        self.db_manager = DatabaseManager()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecogniser()
        self.file_encryptor = FileEncryptor()
        self.ui = UserInterface()
        
        # Directory for storing registered faces
        self.faces_dir = "registered_faces"
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
        
        # Load known faces
        face_data = self.db_manager.get_all_faces()
        self.face_recognizer.load_known_faces(face_data)
        
        # Camera settings
        self.frame_width = 640
        self.frame_height = 480
        self.process_every_n_frames = 5  # Process every n frames
        self.frame_count = 0
        
        # Face detection cooldown tracking
        self.last_face_time = 0
        self.encryption_cooldown = 30  # Seconds between encryption checks
    
        # Performance optimization flags
        self.last_processed_time = time.time()
        
        self.last_status = None  # Track last recognition status
        
    def register_face(self):
        passcode = self.ui.get_admin_passcode()
        
        if passcode != self.ui.admin_passcode:
            print("Invalid passcode")
            return
        
        print("Starting camera for face registration...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        registration_success = False
        
        while not registration_success:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detect_faces(frame)
            
            # Draw rectangles around faces
            display_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(display_frame, "Press 'c' to capture face or 'q' to quit", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Faces detected: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Face Registration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                if len(faces) == 1:
                    name = self.ui.get_face_name()
                    if name:
                        # Save the image
                        timestamp = int(time.time())
                        image_path = os.path.join(self.faces_dir, f"{name}_{timestamp}.jpg")
                        cv2.imwrite(image_path, frame)
                        
                        # Add to database
                        self.db_manager.add_face(name, image_path)
                        
                        print(f"Face registered for {name}")
                        registration_success = True
                        
                        # Reload known faces
                        face_data = self.db_manager.get_all_faces()
                        self.face_recognizer.load_known_faces(face_data)
                else:
                    print("Please ensure exactly one face is visible")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_recognition(self, background_mode=False):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        last_process_time = time.time()
        fps = 0
        recognition_results = []  # Initialize outside the loop
        
        print("\nRunning facial recognition...")
        print("Press 'q' to quit and return to menu\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                current_time = time.time()
                elapsed_time = current_time - last_process_time
                if elapsed_time > 0:
                    fps = 1 / elapsed_time
                last_process_time = current_time
                
                self.frame_count += 1
                if (self.frame_count % self.process_every_n_frames == 0):
                    faces = self.face_detector.detect_faces(frame)
                    
                    if len(faces) > 0:
                        recognition_results = self.face_recognizer.recognize_faces(frame)
                        any_face_recognized = any(result['recognized'] for result in recognition_results)
                        
                        if any_face_recognized:
                            if self.last_status != "recognized":
                                self.file_encryptor.decrypt_files()
                                self.last_status = "recognized"
                        else:
                            if self.last_status != "unknown":
                                self.file_encryptor.encrypt_files()
                                self.last_status = "unknown"
                    
                    if not background_mode and recognition_results:  # Only draw if we have results
                        display_frame = self.ui.draw_recognition_results(frame, recognition_results)
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 120, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow('Facial Recognition Security', display_frame)
                    elif not background_mode:  # If no results, show original frame with FPS
                        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow('Facial Recognition Security', frame)
                
                if not background_mode:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            cap.release()
            if not background_mode:
                cv2.destroyAllWindows()
            print("\nReturning to main menu...")
    
    def close(self):
        self.db_manager.close()

def main():
    system = FacialRecognitionSystem()
    
    while True:
        choice = system.ui.display_menu()
        
        if choice == '1':
            system.register_face()
        elif choice == '2':
            system.run_recognition(background_mode=False)
        elif choice == '3':
            system.run_recognition(background_mode=True)
        elif choice == '4':
            print("Exiting program...")
            system.close()
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()