# core/database.py - Year-Based Organization (NO SECTIONS)
import sqlite3
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager

class EnhancedAttendanceDB:
    def __init__(self, db_name='enhanced_attendance.db'):
        self.db_name = db_name
        self.lock = threading.Lock()
        print("Initializing enhanced database...")
        self.init_database()
        print("Enhanced database initialized successfully!")

    @contextmanager
    def get_db_connection(self):
        """Thread-safe database connection context manager"""
        with self.lock:
            conn = sqlite3.connect(self.db_name, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def init_database(self):
        """Initialize database with YEAR-BASED schema (no sections)"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Students table - organized by academic year
            cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT DEFAULT '',
                academic_year TEXT DEFAULT '1st Year',
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sample_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                created_by TEXT DEFAULT 'teacher',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            # Attendance records table
            cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                academic_year TEXT NOT NULL,
                date DATE DEFAULT (DATE('now')),
                time_in TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT 0.0,
                attendance_session_id TEXT,
                status TEXT DEFAULT 'present',
                FOREIGN KEY (student_id) REFERENCES students(id)
            )''')

            # Attendance sessions table
            cursor.execute('''CREATE TABLE IF NOT EXISTS attendance_sessions (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                academic_year TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_students_detected INTEGER DEFAULT 0,
                session_status TEXT DEFAULT 'active',
                session_type TEXT DEFAULT 'manual'
            )''')

            # Timetable table - organized by academic year (NO SECTIONS)
            cursor.execute('''CREATE TABLE IF NOT EXISTS timetable (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_of_week TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                subject TEXT NOT NULL,
                academic_year TEXT DEFAULT '1st Year',
                is_active BOOLEAN DEFAULT 1,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            # Create indexes
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_students_year ON students(academic_year)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_timetable_year ON timetable(academic_year, day_of_week)''')

            conn.commit()

    # ========== STUDENT MANAGEMENT ==========

    def get_all_students_with_details(self):
        """Get all active students organized by academic year"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        id,
                        name,
                        COALESCE(email, '') as email,
                        COALESCE(academic_year, '1st Year') as academic_year,
                        COALESCE(sample_count, 0) as sample_count,
                        registration_date,
                        COALESCE(is_active, 1) as is_active
                    FROM students 
                    WHERE COALESCE(is_active, 1) = 1 
                    ORDER BY academic_year, registration_date DESC
                """)
                
                rows = cursor.fetchall()
                students = []
                
                for row in rows:
                    student = {
                        'id': row['id'],
                        'name': row['name'],
                        'email': row['email'],
                        'academic_year': row['academic_year'],
                        'year_name': row['academic_year'],  # Consistent naming
                        'class_name': row['academic_year'], # For compatibility
                        'sample_count': row['sample_count'],
                        'registration_date': row['registration_date'],
                        'is_active': bool(row['is_active'])
                    }
                    students.append(student)
                
                return students
        except Exception as e:
            print(f"Error in get_all_students_with_details: {e}")
            return []

    def add_student(self, student_id, name, academic_year, email=""):
        """Add a new student to specific academic year"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO students (id, name, email, academic_year) 
                    VALUES (?, ?, ?, ?)
                """, (student_id, name, email, academic_year))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            print(f"Student ID {student_id} already exists")
            return False
        except Exception as e:
            print(f"Error adding student: {e}")
            return False

    def remove_student(self, student_id):
        """Soft delete a student"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE students 
                    SET is_active = 0, last_updated = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (student_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing student: {e}")
            return False

    def update_student_samples(self, student_id, sample_count):
        """Update student's face sample count"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE students 
                    SET sample_count = ?, last_updated = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (sample_count, student_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating student samples: {e}")
            return False

    def get_students_by_year(self, academic_year):
        """Get students from specific academic year"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM students 
                    WHERE academic_year = ? AND is_active = 1
                    ORDER BY name
                """, (academic_year,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting students by year: {e}")
            return []

    # ========== ATTENDANCE MANAGEMENT ==========

    def record_attendance(self, student_id, subject, academic_year, confidence=0.0, session_id=None):
        """Record student attendance with year tracking"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check for duplicate attendance today
                cursor.execute("""
                    SELECT COUNT(*) FROM attendance 
                    WHERE student_id = ? AND subject = ? AND date = DATE('now')
                """, (student_id, subject))
                
                if cursor.fetchone()[0] > 0:
                    return False, "Already marked present today"
                
                # Record attendance
                cursor.execute("""
                    INSERT INTO attendance 
                    (student_id, subject, academic_year, confidence_score, attendance_session_id) 
                    VALUES (?, ?, ?, ?, ?)
                """, (student_id, subject, academic_year, confidence, session_id))
                
                conn.commit()
                return True, "Attendance recorded successfully"
        except Exception as e:
            print(f"Error recording attendance: {e}")
            return False, f"Database error: {str(e)}"

    def get_today_attendance(self):
        """Get today's attendance records"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        s.id as student_id,
                        s.name,
                        s.academic_year,
                        a.subject,
                        a.time_in,
                        a.confidence_score
                    FROM attendance a 
                    JOIN students s ON a.student_id = s.id 
                    WHERE a.date = DATE('now') 
                    ORDER BY a.time_in DESC
                """)
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting today's attendance: {e}")
            return []

    def get_today_attendance_stats(self):
        """Get attendance statistics for today by year"""
        stats = {
            'total_students': 0,
            'today_present': 0,
            'attendance_rate': 0,
            'classes_today': 0,
            'by_year': {}
        }
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Total active students
                cursor.execute("SELECT COUNT(*) FROM students WHERE COALESCE(is_active, 1) = 1")
                stats['total_students'] = cursor.fetchone()[0]
                
                # Students present today (unique)
                cursor.execute("""
                    SELECT COUNT(DISTINCT student_id) FROM attendance 
                    WHERE date = DATE('now')
                """)
                stats['today_present'] = cursor.fetchone()[0]
                
                # Calculate attendance rate
                if stats['total_students'] > 0:
                    stats['attendance_rate'] = round(
                        (stats['today_present'] / stats['total_students']) * 100, 1
                    )
                
                # Unique subjects today
                cursor.execute("""
                    SELECT COUNT(DISTINCT subject) FROM attendance 
                    WHERE date = DATE('now')
                """)
                stats['classes_today'] = cursor.fetchone()[0]
                
                # Stats by academic year
                cursor.execute("""
                    SELECT 
                        s.academic_year,
                        COUNT(DISTINCT s.id) as total,
                        COUNT(DISTINCT a.student_id) as present
                    FROM students s
                    LEFT JOIN attendance a ON s.id = a.student_id AND a.date = DATE('now')
                    WHERE s.is_active = 1
                    GROUP BY s.academic_year
                """)
                
                for row in cursor.fetchall():
                    year = row['academic_year']
                    stats['by_year'][year] = {
                        'total': row['total'],
                        'present': row['present'] or 0,
                        'rate': round((row['present'] or 0) / row['total'] * 100, 1) if row['total'] > 0 else 0
                    }
                
        except Exception as e:
            print(f"Error getting attendance stats: {e}")
        
        return stats

    def get_attendance_by_date(self, date_str):
        """Get attendance records for specific date with year info"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        s.id as student_id,
                        s.name,
                        s.academic_year,
                        a.subject,
                        a.time_in,
                        a.confidence_score as confidence,
                        a.attendance_session_id as session_id
                    FROM attendance a 
                    JOIN students s ON a.student_id = s.id 
                    WHERE a.date = ? 
                    ORDER BY s.academic_year, a.time_in DESC
                """, (date_str,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting attendance by date: {e}")
            return []

    # ========== SESSION MANAGEMENT ==========

    def create_attendance_session(self, subject, academic_year, duration_minutes=10, session_type='manual'):
        """Create attendance session for specific year"""
        try:
            import uuid
            session_id = str(uuid.uuid4())
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO attendance_sessions (id, subject, academic_year, session_type) 
                    VALUES (?, ?, ?, ?)
                """, (session_id, subject, academic_year, session_type))
                conn.commit()
                return session_id
        except Exception as e:
            print(f"Error creating attendance session: {e}")
            return None

    def get_active_sessions(self):
        """Get active attendance sessions by year"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, subject, academic_year, start_time, session_status, session_type
                    FROM attendance_sessions 
                    WHERE session_status = 'active' 
                    ORDER BY academic_year, start_time DESC
                """)
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting active sessions: {e}")
            return []

    def update_session_stats(self, session_id, students_detected=0):
        """Update session statistics"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE attendance_sessions 
                    SET total_students_detected = ?, 
                        session_status = 'completed',
                        end_time = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (students_detected, session_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating session stats: {e}")
            return False

    def close_session(self, session_id):
        """Close an active session"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE attendance_sessions 
                    SET session_status = 'completed', end_time = CURRENT_TIMESTAMP 
                    WHERE id = ? AND session_status = 'active'
                """, (session_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error closing session: {e}")
            return False

    # ========== TIMETABLE MANAGEMENT (YEAR-BASED) ==========

    def get_current_class(self):
        """Get currently scheduled class for any year"""
        try:
            now = datetime.now()
            current_day = now.strftime('%A').upper()
            current_time = now.strftime('%H:%M')
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT subject, start_time, end_time, academic_year 
                    FROM timetable 
                    WHERE day_of_week = ? 
                      AND start_time <= ? 
                      AND end_time >= ? 
                      AND is_active = 1 
                    ORDER BY start_time 
                    LIMIT 1
                """, (current_day, current_time, current_time))
                
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            print(f"Error getting current class: {e}")
            return None

    def get_upcoming_class(self, minutes_ahead=15):
        """Get next scheduled class within specified minutes"""
        try:
            now = datetime.now()
            current_day = now.strftime('%A').upper()
            current_time = now.strftime('%H:%M')
            future_time = (now + timedelta(minutes=minutes_ahead)).strftime('%H:%M')
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT subject, start_time, end_time, academic_year 
                    FROM timetable 
                    WHERE day_of_week = ? 
                      AND start_time > ? 
                      AND start_time <= ? 
                      AND is_active = 1 
                    ORDER BY start_time 
                    LIMIT 1
                """, (current_day, current_time, future_time))
                
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            print(f"Error getting upcoming class: {e}")
            return None

    def add_timetable_slot(self, day, start_time, end_time, subject, academic_year='1st Year'):
        """Add timetable slot for specific academic year"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO timetable 
                    (day_of_week, start_time, end_time, subject, academic_year) 
                    VALUES (?, ?, ?, ?, ?)
                """, (day.upper(), start_time, end_time, subject, academic_year))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error adding timetable slot: {e}")
            return False

    def get_all_timetable_slots(self):
        """Get all timetable slots organized by year"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, day_of_week, start_time, end_time, subject, academic_year, is_active 
                    FROM timetable 
                    ORDER BY 
                        academic_year,
                        CASE day_of_week 
                            WHEN 'MONDAY' THEN 1
                            WHEN 'TUESDAY' THEN 2
                            WHEN 'WEDNESDAY' THEN 3
                            WHEN 'THURSDAY' THEN 4
                            WHEN 'FRIDAY' THEN 5
                            WHEN 'SATURDAY' THEN 6
                            WHEN 'SUNDAY' THEN 7
                            ELSE 8
                        END, 
                        start_time
                """)
                
                rows = cursor.fetchall()
                slots = []
                
                for row in rows:
                    slot = {
                        'id': row['id'],
                        'day': row['day_of_week'].title(),
                        'start_time': row['start_time'],
                        'end_time': row['end_time'],
                        'subject': row['subject'],
                        'academic_year': row['academic_year'],
                        'is_active': bool(row['is_active'])
                    }
                    slots.append(slot)
                
                return slots
        except Exception as e:
            print(f"Error getting timetable slots: {e}")
            return []

    def delete_timetable_slot(self, timetable_id):
        """Delete a timetable slot"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM timetable WHERE id = ?", (timetable_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting timetable slot: {e}")
            return False

    def toggle_timetable_slot(self, timetable_id):
        """Toggle timetable slot active/inactive"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get current status
                cursor.execute("SELECT is_active FROM timetable WHERE id = ?", (timetable_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Toggle status
                new_status = not bool(row['is_active'])
                cursor.execute("""
                    UPDATE timetable 
                    SET is_active = ? 
                    WHERE id = ?
                """, (new_status, timetable_id))
                conn.commit()
                
                return new_status if cursor.rowcount > 0 else None
        except Exception as e:
            print(f"Error toggling timetable slot: {e}")
            return None

    # ========== UTILITY METHODS ==========

    def get_academic_years(self):
        """Get list of academic years in use"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT academic_year 
                    FROM students 
                    WHERE is_active = 1 
                    ORDER BY academic_year
                """)
                return [row['academic_year'] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting academic years: {e}")
            return ['1st Year', '2nd Year', '3rd Year', '4th Year']

    def get_database_stats(self):
        """Get comprehensive database statistics by year"""
        stats = {'by_year': {}}
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Overall stats
                cursor.execute("SELECT COUNT(*) FROM students WHERE is_active = 1")
                stats['total_students'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM attendance")
                stats['total_attendance_records'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM attendance_sessions WHERE session_status = 'active'")
                stats['active_sessions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM timetable WHERE is_active = 1")
                stats['active_timetable_slots'] = cursor.fetchone()[0]
                
                # Stats by academic year
                cursor.execute("""
                    SELECT 
                        academic_year,
                        COUNT(*) as student_count
                    FROM students 
                    WHERE is_active = 1 
                    GROUP BY academic_year
                    ORDER BY academic_year
                """)
                
                for row in cursor.fetchall():
                    stats['by_year'][row['academic_year']] = {
                        'students': row['student_count']
                    }
                
        except Exception as e:
            print(f"Error getting database stats: {e}")
        
        return stats