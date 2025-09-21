# app.py - FIXED CNN-Enhanced Smart Attendance System

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
import threading
import os
from datetime import datetime, timedelta
from functools import wraps
import atexit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'cnn_enhanced_attendance_system_2024'

logger.info("🚀 Initializing CNN-Enhanced Smart Attendance System...")

# Initialize database first (fast - no blocking)
from enhanced_database import EnhancedAttendanceDB
db = EnhancedAttendanceDB()
logger.info("✅ Database tables created/verified")

# ========== BACKGROUND FACE SYSTEM LOADING ==========
face_system = None
face_system_loading = True
face_system_error = None

def load_face_system():
    """Load face recognition system in background to prevent blocking"""
    global face_system, face_system_loading, face_system_error
    
    try:
        logger.info("🧠 Loading CNN face recognition system in background...")
        from enhanced_face_recognition import EnhancedFaceRecognition
        face_system = EnhancedFaceRecognition()
        
        # Start automatic scheduler
        face_system.start_automatic_scheduler()
        logger.info("📅 Automatic CNN scheduler started")
            
        face_system_loading = False
        logger.info("✅ CNN Face recognition system loaded!")
        
    except Exception as e:
        face_system_loading = False
        face_system_error = str(e)
        logger.error(f"⚠️ Face recognition failed to load: {e}")
        logger.info("📝 System will run without face recognition features")

# Start face system loading in background thread
face_thread = threading.Thread(target=load_face_system, daemon=True)
face_thread.start()

# Cleanup on exit
def cleanup():
    if face_system and hasattr(face_system, 'cleanup'):
        face_system.cleanup()

atexit.register(cleanup)

# ========== HELPER FUNCTIONS ==========
def safe_face_call(method_name, *args, **kwargs):
    """Safely call face system methods with proper error handling"""
    if face_system is None:
        return False, "Face recognition system not ready yet"
    
    if not hasattr(face_system, method_name):
        return False, f"Method {method_name} not available"
    
    try:
        method = getattr(face_system, method_name)
        return method(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {method_name}: {e}")
        return False, f"Error in {method_name}: {str(e)}"

def safe_db_call(method_name, default_value=None, *args, **kwargs):
    """Safely call database methods with fallbacks"""
    if hasattr(db, method_name):
        try:
            method = getattr(db, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database error in {method_name}: {e}")
            return default_value
    else:
        logger.error(f"Database method {method_name} not found")
        return default_value

# Teacher authentication decorator
def teacher_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'teacher_authenticated' not in session:
            return redirect(url_for('teacher_login'))
        return f(*args, **kwargs)
    return decorated_function

# =================== MAIN ROUTES ===================

@app.route('/')
def index():
    """Public dashboard with CNN system status"""
    try:
        # Use safe database calls with fallbacks
        current_class = safe_db_call('get_current_class')
        upcoming_class = safe_db_call('get_upcoming_class', None, 15)
        students = safe_db_call('get_all_students_basic', [])
        if not students:  # Fallback to basic method
            students = safe_db_call('get_all_students', [])
        
        today_stats = safe_db_call('get_today_attendance_stats', {
            'today_present': 0,
            'total_students': len(students) if students else 0,
            'attendance_rate': 0,
            'classes_today': 0
        })
        
        active_sessions = safe_db_call('get_active_sessions', [])
        
        # CNN system status
        if face_system and hasattr(face_system, 'get_system_status'):
            cnn_status = face_system.get_system_status()
        else:
            cnn_status = {
                'system_ready': face_system is not None,
                'camera_status': 'Loading...' if face_system_loading else ('Ready' if face_system else 'Not Available'),
                'model_loaded': face_system is not None,
                'loading': face_system_loading,
                'error': face_system_error,
                'can_recognize': False,
                'total_students': len(students) if students else 0,
                'students_with_training_data': 0
            }
        
        return render_template('public_dashboard.html',
                             current_class=current_class,
                             upcoming_class=upcoming_class,
                             students=students[:10],
                             stats=today_stats,
                             active_sessions=active_sessions,
                             cnn_status=cnn_status)
    except Exception as e:
        logger.error(f"Public dashboard error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    """Teacher authentication"""
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password'].strip()
        
        if email == 'admin@123.com' and password == '123':
            session['teacher_authenticated'] = True
            session['teacher_email'] = email
            flash('Welcome Teacher! CNN-Enhanced system ready.', 'success')
            return redirect(url_for('teacher_dashboard'))
        else:
            flash('Invalid credentials. Use admin@123.com / 123', 'error')
    
    return render_template('teacher_login.html')

@app.route('/teacher/logout')
def teacher_logout():
    """Teacher logout"""
    session.pop('teacher_authenticated', None)
    session.pop('teacher_email', None)
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))

@app.route('/teacher/dashboard')
@teacher_required
def teacher_dashboard():
    """Teacher dashboard with integrated quick start"""
    try:
        current_class = safe_db_call('get_current_class')
        students = safe_db_call('get_all_students_with_details', [])
        if not students:  # Fallback
            students = safe_db_call('get_all_students', [])
            
        timetable = safe_db_call('get_current_week_timetable', [])
        if not timetable:  # Fallback
            timetable = safe_db_call('get_all_timetable_slots', [])
            
        active_sessions = safe_db_call('get_active_sessions', [])
        summary = safe_db_call('get_attendance_summary', {})
        
        # System status
        if face_system and hasattr(face_system, 'get_system_status'):
            cnn_status = face_system.get_system_status()
        else:
            cnn_status = {
                'system_ready': face_system is not None,
                'camera_status': 'Loading...' if face_system_loading else ('Ready' if face_system else 'Not Available'),
                'model_loaded': face_system is not None,
                'loading': face_system_loading,
                'error': face_system_error,
                'can_recognize': False,
                'total_students': len(students) if students else 0,
                'students_with_training_data': 0
            }
        
        # Quick start subjects
        quick_subjects = ['Mathematics', 'Data Science', 'Physics']
        
        return render_template('teacher_dashboard.html',
                             current_class=current_class,
                             students=students,
                             timetable=timetable,
                             active_sessions=active_sessions,
                             summary=summary,
                             cnn_status=cnn_status,
                             quick_subjects=quick_subjects)
    except Exception as e:
        logger.error(f"Teacher dashboard error: {e}")
        return render_template('error.html', error=str(e))

# =================== QUICK START ROUTE ===================

@app.route('/teacher/quick_start', methods=['POST'])
@teacher_required
def quick_start_attendance():
    """Quick start attendance from teacher dashboard"""
    try:
        subject = request.form.get('subject', '').strip()
        custom_subject = request.form.get('custom_subject', '').strip()
        
        # Use custom subject if provided
        if subject == 'custom' and custom_subject:
            subject = custom_subject
        
        if not subject:
            flash('Please select a subject for quick attendance', 'error')
            return redirect(url_for('teacher_dashboard'))
        
        # Check if face system is ready
        if not face_system:
            flash('⚠️ Face recognition system not ready yet. Please wait for system to load completely.', 'warning')
            return redirect(url_for('teacher_dashboard'))
        
        # Start quick attendance session
        success, message = safe_face_call('start_manual_attendance', subject, 10, 'quick_start')
        
        if success:
            flash(f'✅ Quick attendance started for {subject}! Session will run for 10 minutes.', 'success')
        else:
            flash(f'❌ Failed to start attendance: {message}', 'error')
            
    except Exception as e:
        flash(f'Error starting quick attendance: {str(e)}', 'error')
    
    return redirect(url_for('teacher_dashboard'))

# =================== STUDENT MANAGEMENT ===================

@app.route('/teacher/students')
@teacher_required
def manage_students():
    """Student management page"""
    try:
        students = safe_db_call('get_all_students_with_details', [])
        if not students:  # Fallback
            students = safe_db_call('get_all_students', [])
            
        return render_template('student_management.html', students=students)
    except Exception as e:
        logger.error(f"Student management error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/teacher/register_student', methods=['GET', 'POST'])
@teacher_required
def register_student():
    """Register new student with face samples"""
    if request.method == 'POST':
        try:
            student_id = request.form['student_id'].strip()
            name = request.form['name'].strip()
            email = request.form['email'].strip()
            year_name = request.form.get('year_name', 'Default').strip()
            
            if face_system and hasattr(face_system, 'register_student_enhanced'):
                # Register student with CNN face capture
                success, message = face_system.register_student_enhanced(
                    student_id, name, email, year_name, samples_needed=8
                )
            else:
                # Fallback to basic registration
                success = db.add_student(student_id, name, year_name, email)
                if success:
                    message = f"Student {name} registered successfully (Face recognition will be available when system loads)"
                else:
                    message = "Failed to register student"
            
            if success:
                flash(message, 'success')
            else:
                flash(message, 'error')
        except Exception as e:
            flash(f"Registration error: {str(e)}", 'error')
        
        return redirect(url_for('manage_students'))
    
    # Year options for form
    year_options = ['First Year', 'Second Year', 'Third Year', 'Final Year']
    
    return render_template('student_registration.html', year_options=year_options)

@app.route('/teacher/remove_student/<student_id>')
@teacher_required
def remove_student(student_id):
    """Remove student and all data"""
    try:
        success = db.remove_student(student_id)
        
        if success:
            # Clean up training data
            import shutil
            training_dir = f"face_samples/{student_id}"
            if os.path.exists(training_dir):
                shutil.rmtree(training_dir)
            
            flash(f'Student {student_id} removed successfully', 'success')
        else:
            flash(f'Failed to remove student {student_id}', 'error')
    except Exception as e:
        flash(f'Error removing student: {str(e)}', 'error')
    
    return redirect(url_for('manage_students'))

# =================== TIMETABLE MANAGEMENT - FIXED ===================

@app.route('/teacher/timetable', methods=['GET', 'POST'])
@teacher_required
def manage_timetable():
    """Timetable management - FIXED to handle section parameter"""
    if request.method == 'POST':
        try:
            day = request.form['day'].strip()
            start_time = request.form['start_time'].strip()
            end_time = request.form['end_time'].strip()
            subject = request.form['subject'].strip()
            custom_subject = request.form.get('custom_subject', '').strip()
            section = request.form.get('section', 'A').strip()
            
            # Use custom subject if provided
            if subject == 'custom' and custom_subject:
                subject = custom_subject
            
            if not all([day, start_time, end_time, subject]):
                flash('All fields are required', 'error')
                return render_template('timetable_management.html', 
                                     timetable=[], subjects=subjects, sections=sections)
            
            # Check for conflicts
            conflicts = safe_db_call('get_timetable_conflicts', [], day, start_time, end_time)
            if conflicts:
                flash(f'Time conflict with existing {conflicts[0]["subject"]} class', 'warning')
            else:
                # FIXED: Call with section parameter
                success = db.add_timetable_slot(day, start_time, end_time, subject, section)
                if success:
                    flash(f'{subject} class scheduled for {day} {start_time}-{end_time} (Section {section})', 'success')
                else:
                    flash('Failed to add timetable slot', 'error')
        except Exception as e:
            flash(f'Error adding timetable: {str(e)}', 'error')
    
    # Get timetable data
    timetable = safe_db_call('get_full_timetable', [])
    if not timetable:  # Fallback
        timetable = safe_db_call('get_all_timetable_slots', [])
        
    # Subject options
    subjects = [
        'Mathematics', 'Data Science', 'Physics', 'Chemistry', 'Biology', 
        'Computer Science', 'Machine Learning', 'Web Development', 'English', 
        'History', 'Geography', 'Economics', 'Psychology'
    ]
    
    sections = ['A', 'B', 'C', 'D']
    
    return render_template('timetable_management.html',
                         timetable=timetable, subjects=subjects, sections=sections)

@app.route('/teacher/timetable/delete/<int:timetable_id>')
@teacher_required
def delete_timetable_slot(timetable_id):
    """Delete timetable slot"""
    try:
        success = safe_db_call('remove_timetable_slot', False, timetable_id)
        if not success:  # Try alternative method name
            success = safe_db_call('delete_timetable_slot', False, timetable_id)
            
        if success:
            flash('Timetable slot deleted successfully', 'success')
        else:
            flash('Failed to delete timetable slot', 'error')
    except Exception as e:
        flash(f'Error deleting timetable: {str(e)}', 'error')
    
    return redirect(url_for('manage_timetable'))

@app.route('/teacher/timetable/toggle/<int:timetable_id>')
@teacher_required
def toggle_timetable_slot(timetable_id):
    """Toggle timetable slot active status"""
    try:
        # Get current status first
        timetable = safe_db_call('get_full_timetable', [])
        if not timetable:
            timetable = safe_db_call('get_all_timetable_slots', [])
            
        current_slot = next((slot for slot in timetable if slot['id'] == timetable_id), None)
        
        if current_slot:
            new_status = not current_slot.get('is_active', True)
            success = safe_db_call('update_timetable_slot', False, timetable_id, is_active=new_status)
            
            if success:
                status_text = "activated" if new_status else "deactivated"
                flash(f'Timetable slot {status_text} successfully', 'success')
            else:
                flash('Failed to update timetable slot', 'error')
        else:
            flash('Timetable slot not found', 'error')
    except Exception as e:
        flash(f'Error toggling timetable: {str(e)}', 'error')
    
    return redirect(url_for('manage_timetable'))

# =================== ATTENDANCE CONTROL ===================

@app.route('/teacher/start_class/<subject>')
@teacher_required
def start_class(subject):
    """Start manual attendance session"""
    try:
        if not face_system:
            flash('⚠️ Face recognition system not ready yet. Please wait for system to load completely.', 'warning')
            return redirect(url_for('teacher_dashboard'))
        
        # Stop any current session first
        if hasattr(face_system, 'stop_current_session'):
            face_system.stop_current_session()
        
        # Start new CNN-enhanced session
        success, message = safe_face_call('start_manual_attendance', subject, 15, 'manual')
        
        if success:
            flash(f'✅ Started manual attendance for {subject}: {message}', 'success')
        else:
            flash(f'❌ Failed to start class: {message}', 'error')
    except Exception as e:
        flash(f'Error starting class: {str(e)}', 'error')
    
    return redirect(url_for('teacher_dashboard'))

@app.route('/teacher/stop_session')
@teacher_required
def stop_current_session():
    """Stop current attendance session"""
    try:
        success, message = safe_face_call('stop_current_session')
        
        if success:
            flash('Attendance session stopped', 'info')
        else:
            flash(message if message else 'No active session to stop', 'warning')
    except Exception as e:
        flash(f'Error stopping session: {str(e)}', 'error')
    
    return redirect(url_for('teacher_dashboard'))

@app.route('/teacher/train_model')
@teacher_required
def train_cnn_model():
    """Trigger CNN model training"""
    try:
        if not face_system:
            flash('⚠️ Face recognition system not ready yet.', 'warning')
            return redirect(url_for('teacher_dashboard'))
        
        # Start training in background
        if hasattr(face_system, 'train_cnn_model'):
            threading.Thread(target=face_system.train_cnn_model, daemon=True).start()
            flash('🏋️ CNN model training started in background. Check logs for progress.', 'info')
        else:
            flash('❌ Training method not available', 'error')
            
    except Exception as e:
        flash(f'Error starting training: {str(e)}', 'error')
    
    return redirect(url_for('teacher_dashboard'))

# =================== REPORTS ===================

@app.route('/teacher/reports', methods=['GET'])
@teacher_required
def attendance_reports():
    """Attendance reports page"""
    try:
        # Get filter parameters
        selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        selected_subject = request.args.get('subject', 'all')
        
        # Get attendance records
        records = safe_db_call('get_attendance_report_detailed', [], selected_date, selected_subject)
        if not records:  # Fallback method
            records = safe_db_call('get_attendance_by_date', [], selected_date)
        
        # Available subjects for filter
        subjects = [
            'all', 'Mathematics', 'Data Science', 'Physics', 'Chemistry', 'Biology', 
            'Computer Science', 'Machine Learning', 'Web Development', 'English', 
            'History', 'Geography', 'Economics', 'Psychology'
        ]
        
        return render_template('attendance_reports.html',
                             records=records,
                             subjects=subjects,
                             selected_date=selected_date,
                             selected_subject=selected_subject)
    except Exception as e:
        logger.error(f"Reports error: {e}")
        return render_template('error.html', error=str(e))

# =================== SUBJECT MANAGEMENT ===================

@app.route('/teacher/manage_subjects', methods=['GET', 'POST'])
@teacher_required
def manage_subjects():
    """Manage custom subjects"""
    try:
        if request.method == 'POST':
            subject_name = request.form.get('subject_name', '').strip()
            if subject_name:
                flash(f'Subject "{subject_name}" added successfully!', 'success')
            else:
                flash('Subject name is required!', 'error')
        
        # Available subjects
        subjects = [
            'Mathematics', 'Data Science', 'Physics', 'Chemistry', 'Biology', 
            'Computer Science', 'Machine Learning', 'Web Development', 'English', 
            'History', 'Geography', 'Economics', 'Psychology'
        ]
        
        return render_template('subject_management.html', subjects=subjects)
    except Exception as e:
        logger.error(f"Subject management error: {e}")
        return render_template('error.html', error=str(e))

# =================== API ROUTES ===================

@app.route('/api/system_status')
def api_system_status():
    """Get real-time system status"""
    try:
        students = safe_db_call('get_all_students', [])
        today_attendance = safe_db_call('get_today_attendance', [])
        
        today_stats = safe_db_call('get_today_attendance_stats', {
            'today_present': len(today_attendance) if today_attendance else 0,
            'total_students': len(students) if students else 0,
            'attendance_rate': 0,
            'classes_today': 0
        })
        
        # Face system status
        if face_system and hasattr(face_system, 'get_system_status'):
            face_status = face_system.get_system_status()
        else:
            face_status = {
                'system_ready': face_system is not None,
                'loading': face_system_loading,
                'error': face_system_error,
                'can_recognize': False
            }
        
        return jsonify({
            'success': True,
            'face_system': face_status,
            'current_class': safe_db_call('get_current_class'),
            'active_sessions': safe_db_call('get_active_sessions', []),
            'today_stats': today_stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/camera_status')
def api_camera_status():
    """Get camera system status"""
    try:
        if face_system and hasattr(face_system, 'available_cameras'):
            return jsonify({
                'success': True,
                'available_cameras': face_system.available_cameras,
                'current_camera': face_system.camera_index,
                'camera_active': face_system.camera_active
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Face system not ready'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cnn_status')
def api_cnn_status():
    """Get CNN model status"""
    try:
        if face_system:
            status = face_system.get_system_status()
            return jsonify({
                'success': True,
                'model_loaded': status.get('model_loaded', False),
                'embeddings_loaded': status.get('embeddings_loaded', False),
                'can_recognize': status.get('can_recognize', False),
                'total_students': status.get('total_students', 0),
                'students_with_training_data': status.get('students_with_training_data', 0)
            })
        else:
            return jsonify({
                'success': False,
                'loading': face_system_loading,
                'error': face_system_error
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# =================== ERROR HANDLERS ===================

@app.errorhandler(404)
def not_found_error(error):
    try:
        return render_template('error.html', error="Page not found"), 404
    except:
        return "<h1>404 - Page Not Found</h1><p><a href='/'>Go Home</a></p>", 404

@app.errorhandler(500)
def internal_error(error):
    try:
        return render_template('error.html', error="Internal server error"), 500
    except:
        return f"<h1>500 - Server Error</h1><p>{error}</p><p><a href='/'>Go Home</a></p>", 500

# =================== TEMPLATE FILTERS ===================

@app.template_filter('fmt_dt')
def format_datetime(value):
    """Format datetime for display"""
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.strftime('%H:%M:%S')
        except:
            return value
    return value

# =================== MAIN APPLICATION ===================

if __name__ == '__main__':
    logger.info("\n🚀 Starting ENHANCED CNN Smart Attendance System")
    logger.info("=" * 70)
    logger.info("🏠 Public Dashboard: http://localhost:5000")
    logger.info("👨‍🏫 Teacher Login: http://localhost:5000/teacher/login")
    logger.info("🔑 Credentials: admin@123.com / 123")
    logger.info("🧠 CNN Face Recognition: Loading in background...")
    logger.info("")
    logger.info("🆕 ENHANCED FEATURES:")
    logger.info("⚡ Quick Start: Integrated into Teacher Dashboard")
    logger.info("📚 Multiple Subjects: Mathematics, Data Science, Physics, etc.")
    logger.info("🎯 CNN Recognition: TensorFlow-powered face recognition")
    logger.info("📹 Multi-Camera: Automatic camera detection and switching")
    logger.info("⏰ Auto Scheduler: Based on timetable")
    logger.info("🏋️ Model Training: Automatic CNN training after registration")
    logger.info("=" * 70)
    logger.info("✅ Flask server starting immediately (non-blocking!)")
    logger.info("⏳ CNN system will be available once background loading completes")
    
    # Create necessary directories
    directories = [
        'templates', 'models', 'training_data', 'face_samples', 
        'session_logs', 'cnn_logs', 'camera_tests', 'static/css', 'static/js'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)