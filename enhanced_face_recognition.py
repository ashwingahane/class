# enhanced_face_recognition.py - FULL CNN/TensorFlow Integration with Multi-Camera

import cv2
import numpy as np
import os
import time
import threading
import pickle
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFaceRecognition:
    def __init__(self):
        """Initialize Enhanced Face Recognition with CNN capabilities"""
        logger.info("🧠 Initializing Enhanced CNN Face Recognition System...")
        
        # Initialize database connection
        try:
            from enhanced_database import EnhancedAttendanceDB
            self.db = EnhancedAttendanceDB()
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            self.db = None
        
        # CNN Configuration
        self.img_size = (160, 160)  # FaceNet standard
        self.confidence_threshold = 0.85
        self.model_path = 'models/facenet_model.h5'
        self.encoder_path = 'models/face_encoder.pkl'
        self.embeddings_path = 'models/face_embeddings.npy'
        self.labels_path = 'models/face_labels.npy'
        
        # Create directories
        self.create_directories()
        
        # Multi-camera configuration
        self.available_cameras = []
        self.current_camera = None
        self.camera_index = 0
        
        # CNN/TensorFlow components
        self.tf = None
        self.model = None
        self.face_detector = None
        self.label_encoder = None
        self.known_embeddings = None
        self.known_labels = None
        
        # System status
        self.model_loaded = False
        self.ml_loading = False
        self.ml_error = None
        self.camera_active = False
        
        # Threading
        self.training_thread = None
        self.recognition_thread = None
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # Initialize basic components
        self.initialize_face_detector()
        self.scan_available_cameras()
        
        logger.info("✅ Enhanced CNN Face Recognition initialized")
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'models', 'training_data', 'face_samples', 
            'session_logs', 'cnn_logs', 'camera_tests'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.info("📁 Directory structure created")
    
    def initialize_face_detector(self):
        """Initialize OpenCV face detector"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Try to load DNN face detector for better accuracy
            try:
                prototxt_path = 'models/deploy.prototxt'
                weights_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
                
                if os.path.exists(prototxt_path) and os.path.exists(weights_path):
                    self.dnn_detector = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
                    logger.info("✅ DNN face detector loaded")
                else:
                    self.dnn_detector = None
                    logger.info("⚠️ DNN detector files not found, using Haar cascades")
                    
            except Exception as e:
                self.dnn_detector = None
                logger.info(f"⚠️ DNN detector failed: {e}")
            
            logger.info("✅ Face detection system initialized")
            
        except Exception as e:
            logger.error(f"❌ Face detector initialization failed: {e}")
            self.face_cascade = None
            self.dnn_detector = None
    
    def scan_available_cameras(self):
        """Scan for available cameras with multi-camera support"""
        logger.info("📹 Scanning for available cameras...")
        self.available_cameras = []
        
        # Test camera indices 0-4
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Test if camera is actually working
                    height, width = frame.shape[:2]
                    if width > 0 and height > 0:
                        camera_info = {
                            'index': i,
                            'resolution': f"{width}x{height}",
                            'status': 'working'
                        }
                        self.available_cameras.append(camera_info)
                        logger.info(f"✅ Camera {i}: {width}x{height}")
                    
                cap.release()
                
            except Exception as e:
                logger.info(f"❌ Camera {i}: {e}")
        
        if self.available_cameras:
            self.camera_index = self.available_cameras[0]['index']
            logger.info(f"🎯 Primary camera set to index {self.camera_index}")
        else:
            logger.warning("⚠️ No working cameras found!")
    
    def get_next_camera(self):
        """Switch to next available camera"""
        if len(self.available_cameras) > 1:
            current_idx = next((i for i, cam in enumerate(self.available_cameras) 
                              if cam['index'] == self.camera_index), 0)
            next_idx = (current_idx + 1) % len(self.available_cameras)
            self.camera_index = self.available_cameras[next_idx]['index']
            logger.info(f"🔄 Switched to camera {self.camera_index}")
            return True
        return False
    
    def initialize_camera(self, action_name="operation"):
        """Initialize camera with multi-camera fallback"""
        logger.info(f"📹 Initializing camera for {action_name}...")
        
        # Release existing camera
        if self.current_camera:
            try:
                self.current_camera.release()
            except:
                pass
            self.current_camera = None
        
        # Try all available cameras
        for camera_info in self.available_cameras:
            try:
                index = camera_info['index']
                logger.info(f"🔍 Trying camera {index}...")
                
                cap = cv2.VideoCapture(index)
                
                # Set optimal parameters
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                
                # Test capture
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.current_camera = cap
                    self.camera_active = True
                    self.camera_index = index
                    logger.info(f"✅ Camera {index} initialized successfully!")
                    return cap
                else:
                    cap.release()
                    
            except Exception as e:
                logger.error(f"❌ Camera {index} failed: {e}")
        
        logger.error("❌ No working camera available!")
        return None
    
    def load_tensorflow_model(self):
        """Load TensorFlow/Keras model with proper error handling"""
        if self.model_loaded or self.ml_loading:
            return self.model_loaded
        
        self.ml_loading = True
        logger.info("🧠 Loading TensorFlow CNN model...")
        
        try:
            # Import TensorFlow
            import tensorflow as tf
            self.tf = tf
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            # Check if model exists
            if os.path.exists(self.model_path):
                logger.info("📥 Loading existing CNN model...")
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("✅ Pre-trained CNN model loaded")
            else:
                logger.info("🏗️ Creating new CNN model...")
                self.model = self.create_cnn_model()
                logger.info("✅ New CNN model created")
            
            # Load or create label encoder
            if os.path.exists(self.encoder_path):
                with open(self.encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("✅ Label encoder loaded")
            else:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                logger.info("✅ New label encoder created")
            
            # Load embeddings if available
            self.load_face_embeddings()
            
            self.model_loaded = True
            self.ml_error = None
            logger.info("🎉 CNN/TensorFlow system fully loaded!")
            return True
            
        except ImportError as e:
            self.ml_error = f"TensorFlow not installed: {e}"
            logger.error(f"❌ {self.ml_error}")
            return False
        except Exception as e:
            self.ml_error = f"Model loading error: {e}"
            logger.error(f"❌ {self.ml_error}")
            return False
        finally:
            self.ml_loading = False
    
    def create_cnn_model(self):
        """Create CNN model for face recognition"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            # FaceNet-inspired architecture
            model = models.Sequential([
                # Input layer
                layers.Input(shape=(*self.img_size, 3)),
                
                # Convolutional blocks
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Global pooling and dense layers
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                
                # Output layer (will be adjusted based on number of students)
                layers.Dense(128, activation='softmax')  # Embedding size
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("🏗️ CNN model architecture created")
            return model
            
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            return None
    
    def detect_faces_dnn(self, frame):
        """Detect faces using DNN (more accurate)"""
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.dnn_detector.setInput(blob)
            detections = self.dnn_detector.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Convert to (x, y, w, h) format
                    faces.append((x1, y1, x2-x1, y2-y1))
            
            return faces
            
        except Exception as e:
            logger.error(f"DNN face detection error: {e}")
            return []
    
    def detect_faces(self, frame):
        """Detect faces using best available method"""
        if self.dnn_detector is not None:
            faces = self.detect_faces_dnn(frame)
            if faces:
                return faces
        
        # Fallback to Haar cascade
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(60, 60),
                    maxSize=(400, 400)
                )
                return faces
            except Exception as e:
                logger.error(f"Haar cascade detection error: {e}")
        
        return []
    
    def preprocess_face(self, face_img):
        """Preprocess face for CNN model"""
        try:
            # Resize to model input size
            face_resized = cv2.resize(face_img, self.img_size)
            
            # Normalize pixel values
            face_normalized = face_resized.astype('float32') / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
            
        except Exception as e:
            logger.error(f"Face preprocessing error: {e}")
            return None
    
    def extract_face_embedding(self, face_img):
        """Extract face embedding using CNN model"""
        if not self.model_loaded:
            if not self.load_tensorflow_model():
                return None
        
        try:
            # Preprocess face
            face_batch = self.preprocess_face(face_img)
            if face_batch is None:
                return None
            
            # Extract embedding
            embedding = self.model.predict(face_batch, verbose=0)
            return embedding[0]  # Remove batch dimension
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None
    
    def recognize_face(self, face_img):
        """Recognize face using CNN embeddings"""
        if not self.model_loaded or self.known_embeddings is None:
            return None, 0.0
        
        try:
            # Extract embedding for input face
            face_embedding = self.extract_face_embedding(face_img)
            if face_embedding is None:
                return None, 0.0
            
            # Calculate cosine similarities with known embeddings
            similarities = []
            for known_embedding in self.known_embeddings:
                # Cosine similarity
                dot_product = np.dot(face_embedding, known_embedding)
                norm_a = np.linalg.norm(face_embedding)
                norm_b = np.linalg.norm(known_embedding)
                similarity = dot_product / (norm_a * norm_b)
                similarities.append(similarity)
            
            # Find best match
            max_similarity = max(similarities)
            if max_similarity > self.confidence_threshold:
                best_match_idx = similarities.index(max_similarity)
                student_id = self.known_labels[best_match_idx]
                return student_id, max_similarity
            
            return None, max_similarity
            
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            return None, 0.0
    
    def load_face_embeddings(self):
        """Load pre-computed face embeddings"""
        try:
            if os.path.exists(self.embeddings_path) and os.path.exists(self.labels_path):
                self.known_embeddings = np.load(self.embeddings_path)
                self.known_labels = np.load(self.labels_path)
                logger.info(f"✅ Loaded {len(self.known_embeddings)} face embeddings")
                return True
            else:
                logger.info("🔄 No existing embeddings found, will generate on first training")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {e}")
            return False
    
    def train_cnn_model(self):
        """Train CNN model with collected face samples"""
        if not self.load_tensorflow_model():
            return False, "TensorFlow not available"
        
        logger.info("🚀 Starting CNN model training...")
        
        try:
            # Get all students with face samples
            students = self.db.get_all_students() if self.db else []
            
            if not students:
                return False, "No students found in database"
            
            # Collect training data
            training_faces = []
            training_labels = []
            
            for student in students:
                student_id = student['id']
                sample_dir = os.path.join('face_samples', student_id)
                
                if os.path.exists(sample_dir):
                    sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
                    
                    for sample_file in sample_files:
                        sample_path = os.path.join(sample_dir, sample_file)
                        
                        try:
                            # Load and preprocess image
                            img = cv2.imread(sample_path)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                face_batch = self.preprocess_face(img_rgb)
                                
                                if face_batch is not None:
                                    training_faces.append(face_batch[0])  # Remove batch dimension
                                    training_labels.append(student_id)
                                    
                        except Exception as e:
                            logger.error(f"Error processing {sample_path}: {e}")
            
            if len(training_faces) < 10:
                return False, f"Insufficient training data: {len(training_faces)} samples"
            
            # Convert to numpy arrays
            X_train = np.array(training_faces)
            
            # Encode labels
            self.label_encoder.fit(training_labels)
            y_encoded = self.label_encoder.transform(training_labels)
            
            # Convert to categorical
            from tensorflow.keras.utils import to_categorical
            num_classes = len(np.unique(y_encoded))
            y_categorical = to_categorical(y_encoded, num_classes)
            
            # Update model output layer
            if self.model.layers[-1].units != num_classes:
                # Rebuild model with correct output size
                self.model = self.create_cnn_model()
                # Update last layer
                self.model.pop()  # Remove last layer
                self.model.add(self.tf.keras.layers.Dense(num_classes, activation='softmax'))
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            # Train model
            logger.info(f"🏋️ Training with {len(X_train)} samples, {num_classes} students")
            
            history = self.model.fit(
                X_train, y_categorical,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Save model and encoder
            self.model.save(self.model_path)
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Generate and save embeddings for faster recognition
            embeddings = []
            labels = []
            
            for i, (face, label) in enumerate(zip(training_faces, training_labels)):
                embedding = self.extract_face_embedding(np.expand_dims(face, 0))
                if embedding is not None:
                    embeddings.append(embedding)
                    labels.append(label)
            
            if embeddings:
                self.known_embeddings = np.array(embeddings)
                self.known_labels = np.array(labels)
                
                np.save(self.embeddings_path, self.known_embeddings)
                np.save(self.labels_path, self.known_labels)
            
            # Log training results
            final_accuracy = history.history['accuracy'][-1]
            if self.db:
                self.db.log_cnn_training(
                    num_classes, len(X_train), final_accuracy, 
                    self.model_path, len(history.history['accuracy']) * 10  # Approximate training time
                )
            
            logger.info(f"🎉 Training completed! Accuracy: {final_accuracy:.3f}")
            return True, f"Model trained successfully with {final_accuracy:.1%} accuracy"
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False, f"Training error: {str(e)}"
    
    def start_automatic_scheduler(self):
        """Start automatic attendance scheduler based on timetable"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("⏰ Automatic scheduler started")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.scheduler_running:
            try:
                if self.db:
                    current_class = self.db.get_current_class()
                    if current_class:
                        # Check if session already active
                        active_sessions = self.db.get_active_sessions()
                        current_subject = current_class['subject']
                        
                        # Check if session for this subject is already running
                        subject_active = any(s['subject'] == current_subject for s in active_sessions)
                        
                        if not subject_active:
                            logger.info(f"🚀 Auto-starting attendance for {current_subject}")
                            self.start_automatic_attendance(current_subject, duration_minutes=30)
                
                # Check every 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def start_automatic_attendance(self, subject, duration_minutes=10, session_type='automatic'):
        """Start automatic attendance session"""
        return self.start_manual_attendance(subject, duration_minutes, session_type)
    
    def start_manual_attendance(self, subject, duration_minutes=10, session_type='manual'):
        """Start attendance session with full CNN recognition"""
        try:
            # Load CNN model if not loaded
            if not self.model_loaded:
                model_loaded = self.load_tensorflow_model()
                if not model_loaded:
                    logger.warning("⚠️ CNN model not available, using basic face detection")
            
            # Initialize camera
            camera = self.initialize_camera(f"attendance for {subject}")
            if not camera:
                return False, "Camera initialization failed"
            
            logger.info(f"🎯 Starting {session_type} attendance for {subject} ({duration_minutes} minutes)")
            
            # Create session in database
            session_id = f"{session_type}_{subject}_{int(time.time())}"
            if self.db:
                self.db.add_session(session_id, subject, session_type)
            
            start_time = time.time()
            detected_faces = 0
            recognized_students = set()
            total_detections = 0
            
            # Main recognition loop
            while True:
                ret, frame = camera.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Check time limit
                elapsed = time.time() - start_time
                if elapsed > duration_minutes * 60:
                    logger.info("⏰ Time limit reached")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                total_detections += len(faces)
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    color = (0, 255, 0)  # Default green
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if self.model_loaded and self.known_embeddings is not None:
                        # Try CNN recognition
                        student_id, confidence = self.recognize_face(face_roi)
                        
                        if student_id and confidence > self.confidence_threshold:
                            if student_id not in recognized_students:
                                # Mark attendance
                                if self.db:
                                    success = self.db.add_attendance(
                                        student_id, subject, confidence, session_id, 'cnn'
                                    )
                                    if success:
                                        recognized_students.add(student_id)
                                        logger.info(f"✅ CNN Recognition: {student_id} ({confidence:.3f})")
                                
                                # Green box for recognized
                                color = (0, 255, 0)
                                cv2.putText(frame, f"{student_id} ({confidence:.2f})", 
                                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            else:
                                # Yellow for already marked
                                color = (0, 255, 255)
                                cv2.putText(frame, f"{student_id} (Marked)", 
                                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            # Red for unknown
                            color = (0, 0, 255)
                            cv2.putText(frame, f"Unknown ({confidence:.2f})", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        # Basic detection only
                        color = (255, 255, 0)
                        cv2.putText(frame, "Face Detected", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Update rectangle color
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display session info
                remaining = int(duration_minutes * 60 - elapsed)
                info_text = [
                    f"Subject: {subject} ({session_type})",
                    f"Time: {remaining//60}:{remaining%60:02d}",
                    f"Camera: {self.camera_index}",
                    f"Faces: {len(faces)}",
                    f"Recognized: {len(recognized_students)}",
                    f"Total Detections: {total_detections}"
                ]
                
                if self.model_loaded:
                    info_text.append("CNN: ACTIVE")
                else:
                    info_text.append("CNN: LOADING...")
                
                # Draw info
                for i, text in enumerate(info_text):
                    y_pos = 30 + i * 25
                    cv2.putText(frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow(f"CNN Attendance: {subject}", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Session stopped by user")
                    break
                elif key == ord('c'):
                    self.get_next_camera()
                    camera.release()
                    camera = self.initialize_camera(f"attendance for {subject}")
                    if not camera:
                        break
                elif key == ord('t'):
                    # Trigger training
                    logger.info("🏋️ Triggering CNN model training...")
                    threading.Thread(target=self.train_cnn_model, daemon=True).start()
            
            # Cleanup
            camera.release()
            cv2.destroyAllWindows()
            
            # End session
            if self.db:
                self.db.end_session(session_id, len(recognized_students), total_detections)
            
            result_msg = f"Session completed: {len(recognized_students)} students recognized, {total_detections} face detections"
            logger.info(f"🏁 {result_msg}")
            return True, result_msg
            
        except Exception as e:
            # Cleanup on error
            if 'camera' in locals() and camera:
                camera.release()
            cv2.destroyAllWindows()
            
            logger.error(f"❌ Attendance session error: {e}")
            return False, f"Error: {str(e)}"
    
    def register_student_enhanced(self, student_id, name, email, year_name, samples_needed=8):
        """Register student with CNN-enhanced face capture"""
        try:
            # Add student to database
            if self.db:
                success = self.db.add_student(student_id, name, year_name, email)
                if not success:
                    logger.info(f"Student {student_id} already exists in database")
            
            # Initialize camera
            camera = self.initialize_camera(f"registering {name}")
            if not camera:
                return False, "Camera initialization failed"
            
            # Create sample directory
            sample_dir = os.path.join('face_samples', student_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            samples_captured = 0
            quality_scores = []
            
            logger.info(f"📸 Registering {name} - Press SPACE to capture, C to switch camera, Q to quit")
            
            while samples_captured < samples_needed:
                ret, frame = camera.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Show largest face with quality indicator
                best_face = None
                best_quality = 0
                
                for (x, y, w, h) in faces:
                    # Calculate quality score
                    face_area = w * h
                    aspect_ratio = w / h if h > 0 else 0
                    center_score = 1.0 - abs((x + w/2) - frame.shape[1]/2) / (frame.shape[1]/2)
                    
                    quality = (face_area / 10000) * (1.0 if 0.8 < aspect_ratio < 1.25 else 0.5) * center_score
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_face = (x, y, w, h)
                
                # Draw face detection
                if best_face:
                    x, y, w, h = best_face
                    
                    # Color based on quality
                    if best_quality > 0.7:
                        color = (0, 255, 0)  # Green - good
                        quality_text = "EXCELLENT"
                    elif best_quality > 0.5:
                        color = (0, 255, 255)  # Yellow - okay
                        quality_text = "GOOD"
                    else:
                        color = (0, 0, 255)  # Red - poor
                        quality_text = "POOR"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame, f"Quality: {quality_text}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display registration info
                info_text = [
                    f"Registering: {name}",
                    f"Camera: {self.camera_index}",
                    f"Samples: {samples_captured}/{samples_needed}",
                    f"Quality: {best_quality:.2f}" if best_face else "No face detected",
                    "",
                    "Controls:",
                    "SPACE - Capture sample",
                    "C - Switch camera",
                    "Q - Quit registration"
                ]
                
                for i, text in enumerate(info_text):
                    y_pos = 30 + i * 25
                    cv2.putText(frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(f"Registering {name}", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and best_face and best_quality > 0.3:  # Space to capture
                    # Save face sample
                    x, y, w, h = best_face
                    face_roi = frame[y:y+h, x:x+w]
                    
                    sample_path = os.path.join(sample_dir, f"sample_{samples_captured:02d}.jpg")
                    cv2.imwrite(sample_path, face_roi)
                    
                    # Store quality score in database if available
                    if self.db:
                        try:
                            # Create dummy encoding for database
                            dummy_encoding = np.random.random(128)
                            self.db.add_face_sample(student_id, samples_captured, dummy_encoding, best_quality, sample_path)
                        except Exception as e:
                            logger.warning(f"Could not save to database: {e}")
                    
                    quality_scores.append(best_quality)
                    samples_captured += 1
                    logger.info(f"📸 Captured sample {samples_captured}/{samples_needed} (quality: {best_quality:.2f})")
                    
                    # Brief pause to avoid duplicate captures
                    time.sleep(0.5)
                    
                elif key == ord('c'):  # Switch camera
                    if self.get_next_camera():
                        camera.release()
                        camera = self.initialize_camera(f"registering {name}")
                        if not camera:
                            break
                    
                elif key == ord('q'):  # Quit
                    logger.info("Registration cancelled by user")
                    break
            
            # Cleanup
            camera.release()
            cv2.destroyAllWindows()
            
            if samples_captured >= samples_needed // 2:
                avg_quality = np.mean(quality_scores) if quality_scores else 0
                
                # Trigger model training if enough students
                if samples_captured >= samples_needed - 2:
                    logger.info("🏋️ Triggering CNN model training in background...")
                    training_thread = threading.Thread(target=self.train_cnn_model, daemon=True)
                    training_thread.start()
                
                result_msg = f"Registration completed! {samples_captured} samples captured for {name} (avg quality: {avg_quality:.2f})"
                logger.info(f"✅ {result_msg}")
                return True, result_msg
            else:
                return False, f"Insufficient samples: {samples_captured}/{samples_needed}"
                
        except Exception as e:
            if 'camera' in locals() and camera:
                camera.release()
            cv2.destroyAllWindows()
            logger.error(f"❌ Registration error: {e}")
            return False, f"Registration error: {str(e)}"
    
    def stop_current_session(self):
        """Stop current session"""
        try:
            if self.current_camera:
                self.current_camera.release()
                self.current_camera = None
                self.camera_active = False
            cv2.destroyAllWindows()
            return True, "Session stopped successfully"
        except Exception as e:
            return False, f"Error stopping session: {str(e)}"
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            total_students = len(self.db.get_all_students()) if self.db else 0
            students_with_samples = 0
            
            if self.db:
                students = self.db.get_all_students()
                students_with_samples = len([s for s in students if s.get('sample_count', 0) > 0])
        except:
            total_students = 0
            students_with_samples = 0
        
        return {
            'system_ready': self.model_loaded,
            'camera_status': f'Camera {self.camera_index}' if self.camera_active else 'Available',
            'model_loaded': self.model_loaded,
            'loading': self.ml_loading,
            'error': self.ml_error,
            'can_recognize': self.model_loaded and self.known_embeddings is not None,
            'camera_ready': len(self.available_cameras) > 0,
            'total_students': total_students,
            'students_with_training_data': students_with_samples,
            'available_cameras': self.available_cameras,
            'ml_available': self.model_loaded,
            'embeddings_loaded': self.known_embeddings is not None,
            'scheduler_running': self.scheduler_running
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.scheduler_running = False
            if self.current_camera:
                self.current_camera.release()
                self.current_camera = None
            cv2.destroyAllWindows()
            logger.info("🧹 Cleanup completed")
        except:
            pass

# Test function
if __name__ == "__main__":
    logger.info("🧪 Testing Enhanced CNN Face Recognition System...")
    try:
        face_system = EnhancedFaceRecognition()
        logger.info("✅ Face recognition system created successfully!")
        
        status = face_system.get_system_status()
        logger.info(f"📊 System status: {status}")
        
        logger.info("✅ Enhanced CNN Face Recognition test completed!")
        
    except Exception as e:
        logger.error(f"❌ Face recognition test failed: {e}")
        import traceback
        traceback.print_exc()