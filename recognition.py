from cgitb import reset
import face_recognition
import cv2
import numpy as np
import glob
import pickle
import datetime
from datetime import datetime
import time

def start_saving_unknown_face():
    temp_detected_face_dictt.update(unknown_face_dictt)
    known_unknown_id.update(ref_unknown_dictt)
    print("saving unknown person...")
    f=open("ref_embed_unknown.pkl","wb")
    pickle.dump(temp_detected_face_dictt,f)
    f.close()
    f=open("ref_name_unknown.pkl","wb")
    pickle.dump(known_unknown_id,f)
    f.close()
    print(known_unknown_id)

def current_time():
    return str(datetime.now().strftime("%d/%m/%Y%H:%M:%S"))


ref_dictt = {}
ref_unknown_dictt = {}
embed_dictt = {}
temp_detected_face_dictt = {}
unknown_face_dictt = {}

try:
    f=open("ref_name.pkl","rb")
    ref_dictt=pickle.load(f)        
    f.close()

    f=open("ref_embed.pkl","rb")
    embed_dictt=pickle.load(f)      
    f.close()

    f=open("ref_name_unknown.pkl","rb")
    ref_unknown_dictt=pickle.load(f)        
    f.close()

    f=open("ref_embed_unknown.pkl","rb")
    unknown_face_dictt = pickle.load(f)
    f.close()
except:
    print("no registered or unknown face found")

known_face_encodings = []    
known_face_id = []  
known_unknown_id = {}

def store_unknown_person(face_encoding):
    unknown_name = "? " + current_time()

    unknown_id = current_time()
    known_face_encodings.append(face_encoding)
    known_face_id.append(unknown_id)
    ref_dictt[unknown_id] = unknown_name
    known_unknown_id[unknown_id] = unknown_name
    temp_detected_face_dictt[unknown_id] = [face_encoding]

for ref_id , embed_list in unknown_face_dictt.items():
        for my_embed in embed_list:
            known_face_encodings += [my_embed]
            known_face_id += [ref_id]
            ref_dictt[ref_id] = ref_unknown_dictt[ref_id]

for ref_id , embed_list in embed_dictt.items():
    for my_embed in embed_list:
        known_face_encodings += [my_embed]
        known_face_id += [ref_id]

video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True  :
  
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_id = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            id = ""

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = -1
            if known_face_encodings:
                best_match_index = np.argmin(face_distances)

            if best_match_index != -1 and matches[best_match_index]:
                id = known_face_id[best_match_index]
            face_id.append(id)


    process_this_frame = not process_this_frame

    if face_locations:
        for (top_s, right, bottom, left), id, face_encoding in zip(face_locations, face_id, face_encodings):
            top_s *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top_s), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            try:
                is_not_exist = not id
                if is_not_exist == False:
                    writtenName = ref_dictt[id]
                else:
                    store_unknown_person(face_encoding)
                    writtenName = "unknown person"

                cv2.putText(frame, writtenName, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            except Exception as e:
                print("err "+ str(e))
           

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
start_saving_unknown_face()
cv2.destroyAllWindows()