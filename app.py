import os
import csv
import cv2
import shutil
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from flask import Flask,render_template,redirect,request,session,g,url_for
from datetime import datetime, date
import joblib
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask_mysqldb import MySQL
import MySQLdb.cursors

load_dotenv()
# Initializing the flask app

app=Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Flask error handler
@app.errorhandler(404)
@app.errorhandler(401)
@app.errorhandler(500)
def http_error_handler(error):
    return render_template('Error.html')

# MySQL configurations
app.config['MYSQL_HOST'] = os.getenv("MYSQL_HOST")
app.config['MYSQL_USER'] = os.getenv("MYSQL_USER")
app.config['MYSQL_PASSWORD'] = os.getenv("MYSQL_PASSWORD")
app.config['MYSQL_DB'] = os.getenv("MYSQL_DB")

mysql = MySQL(app)
    
# Flask assign admin
@app.before_request
def before_request():
    g.user = None
    if 'admin' in session:
        g.user = session['admin']

# Current Date & Time
datetoday = date.today().strftime("%d-%m-%Y")
datetoday2 = date.today().strftime("%d %B %Y")

# Capture the video
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# ======= Check and Make Folders ========
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('UserList'):
    os.makedirs('UserList')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
        
# ======= Total Registered Users ========
def totalreg():
    return len(os.listdir('static/faces'))


# ======= Get Face From Image =========
def extract_faces(img):
    if img is None:
        return ()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=7)
    return face_points


# Load the CNN model globally once
cnn_model = load_model('static/face_recognition_model.h5')

# Get class names (assuming each folder in 'static/faces' is a user/class)
class_names = sorted([
    d for d in os.listdir('static/faces')
    if os.path.isdir(os.path.join('static/faces', d)) and not d.startswith('.')
])

# ======= Identify Face Using ML ========
def identify_face(face_img):
    # face_img is expected to be a BGR image (from OpenCV)
    if face_img is None:
        return "No face detected"
    # Resize to (224, 224) because your CNN expects this size
    face_img = cv2.resize(face_img, (224, 224))
    
    # Convert BGR to RGB because Keras models usually expect RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixels to [0,1]
    face_img = face_img.astype('float32') / 255.0
    
    # Expand dims to shape (1, 224, 224, 3)
    face_img = np.expand_dims(face_img, axis=0)
    
    # Predict class probabilities
    preds = cnn_model.predict(face_img)
    
    # Get the index of highest probability
    pred_index = np.argmax(preds)
    
     # Debug info
    print("Predictions:", preds)
    print("Predicted index:", pred_index)
    print("Class names:", class_names)
    
    # Get the class name
    # predicted_class = class_names[pred_index]
    
    # return predicted_class
    if 0 <= pred_index < len(class_names):
        return class_names[pred_index]
    else:
        return "Unknown"

# ======= Train Model Using Available Faces ========
def train_model():
    face_dir = 'static/faces'
    if len(os.listdir(face_dir)) == 0:
        return

    # Remove old CNN model if exists
    if 'face_recognition_model.h5' in os.listdir('static'):
        os.remove('static/face_recognition_model.h5')

    # Data generator for training
    datagen = ImageDataGenerator(
        rescale=1/255.,
        validation_split=0.2  # Split training and validation
    )

    train_data = datagen.flow_from_directory(
        face_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        face_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Simple CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(10, kernel_size=3, activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(10, kernel_size=2, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(10, kernel_size=2, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        epochs=5,
        validation_data=val_data
    )

    # Save the model
    model.save('static/face_recognition_model.h5')

# ======= Remove Attendance of Deleted User ======
def remAttendance():

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Collect valid IDs from both user tables
    cur.execute("SELECT id FROM registered_user")
    registered_ids = {str(row['id']) for row in cur.fetchall()}

    cur.execute("SELECT id FROM unregistered_user")
    unregistered_ids = {str(row['id']) for row in cur.fetchall()}

    valid_ids = registered_ids | unregistered_ids

    # If there are valid IDs, remove all attendance records that don't belong to them
    if valid_ids:
        ids_str = ",".join([f"'{i}'" for i in valid_ids])
        cur.execute(f"DELETE FROM attendance WHERE id NOT IN ({ids_str})")

    mysql.connection.commit()
    cur.close()

# ======== Get Info From Attendance File =========
def extract_attendance():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    query = """
        SELECT a.name, a.id, a.section, a.time,
            CASE
                WHEN a.id IN (SELECT id FROM unregistered_user) THEN 'Unregistered'
                WHEN a.id IN (SELECT id FROM registered_user) THEN 'Registered'
                ELSE 'Unknown'
            END AS status
        FROM attendance a
        WHERE DATE(a.time) = %s
    """
    datetoday_mysql = date.today().strftime("%Y-%m-%d")
    cur.execute(query, (datetoday_mysql,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return [], [], [], [], datetoday, [], 0

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    sec   = [r['section'] for r in rows]
    times = [r['time'].strftime("%H:%M:%S") for r in rows]  # just time
    reg   = [r['status'] for r in rows]
    l     = len(rows)

    return names, rolls, sec, times, datetoday, reg, l

# ======== Save Attendance =========
def add_attendance(name):
    username, userid, usersection = name.split('$')
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Check if already marked today (ignoring time, only by DATE)
    cur.execute("""
        SELECT * FROM attendance 
        WHERE id=%s AND DATE(time)=%s
    """, (userid, datetime.now().strftime("%Y-%m-%d")))
    already = cur.fetchone()

    if already:
        cur.close()
        return 

    # Insert new attendance with full date+time
    cur.execute("""
        INSERT INTO attendance (id, name, section, time)
        VALUES (%s, %s, %s, %s)
    """, (userid, username, usersection, current_datetime))
    mysql.connection.commit()
    cur.close()

# ======= Flask Home Page =========
@app.route('/')
def home():
    if g.user:
        return render_template('HomePage.html', admin=True, mess='Logged in as Administrator', user=session['admin'])

    return render_template('HomePage.html', admin=False, datetoday2=datetoday2)

@app.route('/attendance')
def take_attendance():
    # Fetch today's attendance from MySQL
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    
    return render_template(
        'Attendance.html',
        names=names,
        rolls=rolls,
        sec=sec,
        times=times,
        l=l,
        datetoday2=datetoday2
    )
    
    
@app.route('/attendancebtn', methods=['GET'])
def attendancebtn():
    if len(os.listdir('static/faces')) == 0:
        return render_template('Attendance.html', datetoday2=datetoday2,
                               mess='Database is empty! Register yourself first.')

    if 'face_recognition_model.h5' not in os.listdir('static'):
        train_model()

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    ret = True
    j = 1
    flag = None

    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)  # Detect all faces

        identified_person_name = "Unknown"
        identified_person_id = "N/A"

        if faces is not None and len(faces) > 0:
            (x, y, w, h) = faces[0]  # Use first detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)

            # Resize and prepare the face
            face_img = cv2.resize(frame[y:y + h, x:x + w], (224, 224))

            identified_person = identify_face(face_img)

            if identified_person is not None and '$' in identified_person:
                identified_person_name, identified_person_id, *_ = identified_person.split('$')

                if flag != identified_person:
                    j = 1
                    flag = identified_person

                if j % 20 == 0:
                    add_attendance(identified_person)
            else:
                identified_person = "Unknown"

            cv2.putText(frame, f'Name: {identified_person_name}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, f'ID: {identified_person_id}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Press Esc to close', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 127, 255), 2, cv2.LINE_AA)

            j += 1
        else:
            j = 1
            flag = None

        # Display the frame
        cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Attendance', 800, 600)
        cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)

@app.route('/adduser')
def add_user():
    return render_template('AddUser.html')

@app.route('/adduserbtn', methods=['GET', 'POST'])
def adduserbtn():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newusersection = request.form['newusersection']

    # Open camera
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        return render_template('AddUser.html', mess='Camera not available.')

    # Create user folder for storing images
    userimagefolder = f'static/faces/{newusername}${newuserid}${newusersection}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    # Check if user already exists in DB
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM unregistered_user WHERE id = %s", (newuserid,))
    existing_user = cur.fetchone()
    if existing_user:
        cap.release()
        cur.close()
        return render_template('AddUser.html', mess='User already exists in database.')

    images_captured = 0
    frame_count = 0
    max_frames = 1000

    while images_captured < 50 and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if faces is None or len(faces) == 0:
            cv2.putText(frame, 'No face detected. Please face the camera.', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                cv2.putText(frame, f'Images Captured: {images_captured}/50', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
                if frame_count % 10 == 0:
                    img_name = f'{newusername}_{images_captured}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, img_name), frame[y:y + h, x:x + w])
                    images_captured += 1

        cv2.namedWindow('Collecting Face Data', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Collecting Face Data', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Collecting Face Data', frame)

        if cv2.waitKey(1) == 27:  # ESC to exit
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    if images_captured == 0:
        shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='Failed to capture photos.')

    # Insert new user into MySQL
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("""
        INSERT INTO unregistered_user (name, id, section)
        VALUES (%s, %s, %s)
    """, (newusername, newuserid, newusersection))
    mysql.connection.commit()
    cur.close()

    # Retrain model after adding user
    train_model()

    # Fetch updated unregistered user list
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT name, id, section FROM unregistered_user")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [str(row['id']) for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)

    return render_template('UnregisterUserList.html',
                           names=names, rolls=rolls, sec=sec, l=l,
                           mess=f'Number of Unregistered Students: {l}')

@app.route('/attendancelist')
def attendance_list():
    if not g.user:
        return render_template('LogInForm.html')

    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('AttendanceList.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg,
                           l=l)
    
# ========== Flask Search Attendance by Date ============
@app.route('/attendancelistdate', methods=['GET', 'POST'])
def attendancelistdate():
    if not g.user:
        return render_template('LogInForm.html')

    date_selected = request.form['date']  # "YYYY-MM-DD"

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("""
        SELECT a.name, a.id, a.section, a.time,
            CASE
                WHEN a.id IN (SELECT id FROM unregistered_user) THEN 'Unregistered'
                WHEN a.id IN (SELECT id FROM registered_user) THEN 'Registered'
                ELSE 'Unknown'
            END AS status
        FROM attendance a
        WHERE DATE(a.time) = %s
    """, (date_selected,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return render_template('AttendanceList.html', names=[], rolls=[], sec=[], times=[], reg=[], l=0,
                               mess="No records for this date.")

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    sec = [r['section'] for r in rows]
    times = [r['time'].strftime("%H:%M:%S") for r in rows]
    dates = [row['time'] for row in rows]
    reg = [r['status'] for r in rows]
    l = len(rows)

    return render_template('AttendanceList.html',
                        names=names, rolls=rolls, sec=sec,
                        times=times, dates=dates, reg=reg,
                        l=l, mess=f"Total Attendance: {l}")

# ========== Flask Search Attendance by ID ============
@app.route('/attendancelistid', methods=['GET', 'POST'])
def attendancelistid():
    if not g.user:
        return render_template('LogInForm.html')

    student_id = request.form.get('id')
    if not student_id:
        return render_template('AttendanceList.html', names=[], rolls=[], sec=[], times=[], dates=[], reg=[], l=0, mess="No ID provided!")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM attendance WHERE id = %s", (student_id,))
    rows = cur.fetchall()
    cur.close()

    if rows:
        names = [row['name'] for row in rows]
        rolls = [row['id'] for row in rows]
        sec   = [row['section'] for row in rows]
        times = [row['time'].strftime("%H:%M:%S") if row['time'] else "N/A" for row in rows]
        dates = [row['time'].strftime("%Y-%m-%d") if row['time'] else "N/A" for row in rows]
        reg   = ['Registered' if row['id'] in [r['id'] for r in rows] else 'Unregistered' for row in rows]
        l = len(rows)
        return render_template('AttendanceList.html',
                               names=names, rolls=rolls, sec=sec,
                               times=times, dates=dates, reg=reg,
                               l=l, mess=f"Total Attendance: {l}")
    else:
        return render_template('AttendanceList.html',
                               names=[], rolls=[], sec=[],
                               times=[], dates=[], reg=[],
                               l=0, mess="Nothing Found!")

# ========== Flask Unregister a User ============
@app.route('/unregisteruser', methods=['POST'])
def unregisteruser():
    if not g.user:
        return render_template('LogInForm.html')

    # remove_empty_cells()
    try:
        idx = int(request.form['index'])
    except (ValueError, KeyError):
        return "Invalid index (not a number or missing)", 400

     # Get the user to unregister from DB
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, name, user_id, section FROM registered_users ORDER BY id ASC")
    registered = cur.fetchall()
    if idx < 0 or idx >= len(registered):
        return "Invalid index", 400

    user = registered[idx]  # tuple: (id, name, user_id, section)

    # Move the face folder
    old_folder = f"static/faces/{user[1]}${user[2]}${user[3]}"
    new_folder = f"static/faces/{user[1]}${user[2]}$None"
    if os.path.exists(old_folder):
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        shutil.move(old_folder, new_folder)

    # Insert into unregistered_users
    cur.execute(
        "INSERT INTO unregistered_users (name, user_id, section) VALUES (%s, %s, %s)",
        (user[1], user[2], 'None')
    )

    # Delete from registered_users
    cur.execute("DELETE FROM registered_users WHERE id=%s", (user[0],))
    mysql.connection.commit()
    cur.close()

    # Return updated list
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT name, user_id, section FROM registered_users ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [r[0] for r in rows]
    rolls = [r[1] for r in rows]
    sec = [r[2] for r in rows]
    l = len(rows)

    mess = f'Number of Registered Students: {l}' if l > 0 else "Database is empty!"

    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)

# ========== Flask Unregister User List ============
@app.route('/unregisteruserlist')
def unregister_user_list():
    if not g.user:
        return render_template('LogInForm.html')
    
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT name, id, section FROM unregistered_user ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Database is empty!")

    names = [row['name'] for row in rows]
    rolls = [row['id'] for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)

    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=sec, l=l,
                           mess=f'Number of Unregistered Students: {l}')

# ========== Flask Delete a User from Unregistered List ============
@app.route('/deleteunregistereduser', methods=['POST'])
def deleteunregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    idx = int(request.form['index'])

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM unregistered_user ORDER BY id ASC")
    unregistered = cur.fetchall()

    if idx >= len(unregistered):
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid user index.")

    user = unregistered[idx]
    username, userid, usersec = user['name'], user['id'], user['section']

    folder = f'static/faces/{username}${userid}${usersec}'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        train_model()

    cur.execute("DELETE FROM unregistered_user WHERE id = %s", (userid,))
    mysql.connection.commit()
    cur.close()

    return redirect(url_for('unregister_user_list'))
    
# ========== Flask Register User List ============
@app.route('/registeruserlist')
def register_user_list():
    if not g.user:
        return render_template('LogInForm.html')

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    cur.execute("SELECT name, id, section FROM registered_user ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [row['name'] for row in rows]
    rolls = [row['id'] for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)

    mess = f'Number of Registered Students: {l}' if l else "Database is empty!"
    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)
        
# ========== Flask Register a User ============
@app.route('/registeruser', methods=['POST'])
def registeruser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form['index'])
        section = request.form['section']
    except (ValueError, KeyError):
        return "Invalid input", 400

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM unregistered_user ORDER BY id ASC")
    unregistered = cur.fetchall()

    if idx < 0 or idx >= len(unregistered):
        return "Invalid user index", 400

    user = unregistered[idx]
    name, user_id = user['name'], user['id']

    # Move the face folder
    old_folder = f"static/faces/{name}${user_id}$None"
    new_folder = f"static/faces/{name}${user_id}${section}"
    if os.path.exists(old_folder):
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        shutil.move(old_folder, new_folder)

    # Insert into registered_user
    cur.execute(
        "INSERT INTO registered_user (name, id, section) VALUES (%s, %s, %s)",
        (name, user_id, section)
    )

    # Delete from unregistered_user
    cur.execute("DELETE FROM unregistered_user WHERE id=%s", (user_id,))

    # Commit both operations to save changes permanently
    mysql.connection.commit()
    cur.close()

    # Reload unregistered list
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT name, id, section FROM unregistered_user ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    secs = [r['section'] for r in rows]
    l = len(rows)

    mess = f'Number of Unregistered Students: {l}' if l > 0 else "Database is empty!"
    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=secs, l=l, mess=mess)
        
# ========== Flask Delete a User from Registered List ============
@app.route('/deleteregistereduser', methods=['POST'])
def deleteregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    idx = int(request.form['index'])

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM registered_user ORDER BY id ASC")
    registered = cur.fetchall()

    if idx >= len(registered):
        return render_template('RegisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid user index.")

    user = registered[idx]
    username, userid, usersec = user['name'], user['id'], user['section']

    # Delete face folder
    folder = f'static/faces/{username}${userid}${usersec}'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        train_model()

    # Delete from DB
    cur.execute("DELETE FROM registered_user WHERE id = %s", (userid,))
    mysql.connection.commit()
    cur.close()

    # Refresh list
    return redirect(url_for('register_user_list'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        session.pop('admin', None)
        return redirect(url_for('home', admin=False))

    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == '12345':
            session['admin'] = request.form['username']
            
            # Record login time in MySQL
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO admin (username) VALUES (%s)", (session['admin'],))
            mysql.connection.commit()
            cur.close()
            
            
            return redirect(url_for('home', admin=True, mess='Logged in as Administrator'))
        else:
            return render_template('LogInForm.html', mess='Incorrect Username or Password')
    return render_template('LogInForm.html')

# ======== Flask Logout =========
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return render_template('LogInFrom.html')

# ======== Admin Log =========
@app.route('/adminlog')
def adminlog():
    if not g.user:
        return render_template('LogInForm.html')

    cur = mysql.connection.cursor()
    cur.execute("SELECT username, time FROM admin ORDER BY time DESC")
    logs = cur.fetchall()
    cur.close()
    
    usernames = [log[0] for log in logs]
    times = [log[1] for log in logs]

    return render_template('AdminLog.html', usernames=usernames, times=times, l=len(logs))

# Main Function
if __name__ == '__main__':
    app.run(port=5001, debug=True)