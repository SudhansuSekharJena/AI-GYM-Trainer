import cv2 as cv
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle
  
  
# CALCULATE ANGLES FOR SHOULDER PRESS
def calculate_angle_shoulder_press(shoulder, elbow, wrist):
    shoulder = np.array(shoulder)
    elbow = np.array(elbow)
    wrist = np.array(wrist)

    # Calculate the vectors representing the arm segments
    upper_arm_vector = elbow - shoulder
    forearm_vector = wrist - elbow

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(upper_arm_vector, forearm_vector)
    upper_arm_magnitude = np.linalg.norm(upper_arm_vector)
    forearm_magnitude = np.linalg.norm(forearm_vector)

    # Calculate the angle between the upper arm and forearm using the dot product formula
    angle_radians = np.arccos(dot_product / (upper_arm_magnitude * forearm_magnitude))

    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

# -----------------------------------------------------------------

# CALCULATE ANGLES FOR SQUAT

def calculate_angle_squat(hip, knee, ankle):
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    # Calculate the vectors representing the thigh and calf segments
    thigh_vector = knee - hip
    calf_vector = ankle - knee

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(thigh_vector, calf_vector)
    thigh_magnitude = np.linalg.norm(thigh_vector)
    calf_magnitude = np.linalg.norm(calf_vector)

    # Calculate the angle between the thigh and calf using the dot product formula
    angle_radians = np.arccos(dot_product / (thigh_magnitude * calf_magnitude))

    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
  


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv.VideoCapture(0)
    
    # open the camera
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    counter = 0
    stage = None
    exercises = ["bicep_curl", "pushup", "squat", "shoulder_press"]

    while True:
        print("Select an exercise:")
        for i, exercise in enumerate(exercises):
            print(f"{i+1}. {exercise.capitalize()}")
        try:
            user_choice = int(input("Enter your choice (1-4): "))
            if user_choice < 1 or user_choice > len(exercises):
                print("Invalid choice. Please try again.")
                continue
            exercise = exercises[user_choice - 1]
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Create a fullscreen window
        cv.namedWindow('Mediapipe Feed', cv.WINDOW_NORMAL)
        cv.setWindowProperty('Mediapipe Feed', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:  # Check if landmarks are detected
                    landmarks = results.pose_landmarks.landmark
                    
                    #  BICEP CURL
                    if exercise == "bicep_curl":
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        angle = calculate_angle(shoulder, elbow, wrist)
                        if angle > 160:
                            stage = "down"
                        if angle < 30 and stage == 'down':
                            stage="up"
                            counter+=1
                        cv.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

                        
                    #  PUSHUP 
                    elif exercise == "pushup":
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        angle = calculate_angle(shoulder, elbow, wrist)

                        if angle > 160:
                            stage = "up"
                        if angle < 30 and stage == "up":
                            stage = "down"
                            counter += 1
                        cv.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
      
                
                            
                    #  SQUAT 
                    elif exercise == "squat":
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        # angle = calculate_angle(hip, knee, ankle)
                        angle = calculate_angle_squat(hip, knee, ankle)

                        if angle > 160:
                            stage = "up"
                        if angle < 90 and stage == "up":
                            stage = "down"
                            counter += 1
                        cv.putText(image, str(angle), tuple(np.multiply(knee, [640, 480]).astype(int)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

                            
                    #  SHOULDER PRESS
                    elif exercise == "shoulder_press":
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        angle = calculate_angle_shoulder_press(shoulder, elbow, wrist)
                        # angle = calculate_angle(hip, knee, ankle)

                        if angle < 20:
                            stage = "up"
                        if angle > 139 and stage == "up":
                            stage = "down"
                            counter += 1
                        cv.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)


                    # cv.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

            except Exception as e:
                print("Error processing frame:", e)

            cv.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv.putText(image, 'REPS', (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(image, str(counter), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(image, 'STAGE', (65, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(image, stage, (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv.imshow('Mediapipe Feed', image)
            key = cv.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):  # Toggle window size on 'm' key press
                current_state = cv.getWindowProperty('Mediapipe Feed', cv.WND_PROP_FULLSCREEN)
                if current_state == cv.WINDOW_FULLSCREEN:
                    cv.setWindowProperty('Mediapipe Feed', cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
                else:
                    cv.setWindowProperty('Mediapipe Feed', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()

