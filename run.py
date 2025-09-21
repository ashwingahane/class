# run.py - Application Launcher
"""
Smart Attendance System Launcher
Run this file to start the application
"""

import os
import sys

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['cv2', 'numpy', 'flask']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'cv2':
                missing_packages.append('opencv-python')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'templates',
        'static/css', 
        'static/js',
        'face_samples',
        'session_logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created successfully!")

def main():
    """Main launcher function"""
    print("Smart Automatic Attendance System")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Import and run the main application
    try:
        from app import app
        
        print("\nStarting server...")
        print("Open your browser and go to:")
        print("  Public Dashboard: http://localhost:5000")
        print("  Teacher Login: http://localhost:5000/teacher/login")
        print("  Login credentials: admin@123.com / 123")
        print("\nFeatures:")
        print("- Automatic attendance based on timetable")
        print("- Manual attendance sessions")
        print("- Face recognition with 8 samples per student")
        print("- Timetable management with delete/edit")
        print("- Real-time dashboard updates")
        print("\nPress Ctrl+C to stop the server")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"Error importing application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()