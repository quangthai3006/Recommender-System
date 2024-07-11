import mysql.connector
from mysql.connector import Error
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Thay đổi mã hóa mặc định của sys.stdout sang utf-8
sys.stdout.reconfigure(encoding='utf-8')

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

# Thông tin kết nối
host_name = "127.0.0.1"
user_name = "root"
user_password = "thai300622"
db_name = "thaipc"

# Tạo kết nối
connection = create_connection(host_name, user_name, user_password, db_name)

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result, cursor.column_names
    except Error as e:
        print(f"The error '{e}' occurred")

# select_users = "SELECT * FROM doctor_infors"
select_users = """
SELECT doctor_infors.*, users.image, users.firstName, users.lastName, description_alls.description, role_codes.valueVI AS roleValue, position_codes.valueVI AS positionValue, specialties.name AS nameSpecialty, GROUP_CONCAT(DISTINCT FROM_UNIXTIME(schedules.date / 1000) ORDER BY FROM_UNIXTIME(schedules.date / 1000) SEPARATOR ' ') AS dates, 
       GROUP_CONCAT(DISTINCT schedules.timeType ORDER BY schedules.timeType SEPARATOR ' ') AS timeTypes
FROM doctor_infors
JOIN users ON doctor_infors.doctorId = users.id
JOIN description_alls ON doctor_infors.doctorId = description_alls.doctorId
JOIN allcodes AS role_codes ON users.roleId = role_codes.keyMap
JOIN allcodes AS position_codes ON users.positionId = position_codes.keyMap
JOIN schedules ON doctor_infors.doctorId = schedules.doctorId
JOIN specialties ON doctor_infors.specialtyId = specialties.id
WHERE FROM_UNIXTIME(schedules.date / 1000) >= CURDATE()
GROUP BY doctor_infors.doctorId


"""

users, columns = execute_read_query(connection, select_users)

# JOIN schedules ON doctor_infors.doctorId = schedules.doctorId
# WHERE FROM_UNIXTIME(schedules.date / 1000) >= CURDATE()
# WHERE schedules.date >= CURDATE()
# GROUP BY doctor_infors.doctorId, schedules.date
# schedules.date, schedules.timeType
# ,  schedules.date, GROUP_CONCAT(schedules.timeType SEPARATOR ' ') AS timeTypes
# ,  FROM_UNIXTIME(schedules.date / 1000) AS date , GROUP_CONCAT(schedules.timeType SEPARATOR ' ') AS timeTypes




# Chuyển đổi users thành DataFrame
users_df = pd.DataFrame(users, columns=columns)


# Thêm cột combineFeatures
def combine_features(row):
    return f"{row['provinceId']} {row['specialtyId']} {row['priceId']} {row['healthFacilitiesId']} {row['description']} {row['dates']} {row['timeTypes']}   "

# {row['date']} {row['timeType']} 
# {row['combineDateTime']}
users_df['combineFeatures'] = users_df.apply(combine_features, axis=1)
print(users_df)
tf = TfidfVectorizer()
tfMatrix = tf.fit_transform(users_df['combineFeatures'])
print(tfMatrix)


# {row['provinceId']} {row['specialtyId']} {row['priceId']} {row['healthFacilitiesId']} {row['description']}

sililar = cosine_similarity(tfMatrix)
print(sililar)


number = 5

@app.route('/api/recommender-system', methods=['GET'])
def get_data():
    result = []
    productid = request.args.get('id')

    if productid is None:
        return jsonify({'error': 'Product ID is missing'}), 400

    try:
        productid = int(productid)
    except ValueError:
        return jsonify({'error': 'Invalid Product ID'}), 400

    if productid not in users_df['doctorId'].values:
        return jsonify({'error': 'ID không hợp lệ'}), 400

    indexproduct = users_df[users_df['doctorId'] == productid].index[0]

    similarity = list(enumerate(sililar[indexproduct]))

    sortedSimilarity = sorted(similarity, key=lambda x: x[1], reverse=True)

    def getName(index):
        # return users_df[users_df.index == index]['doctorId'].values[0]
        row = users_df.iloc[index]
        return {
            'description': row['description'],
            'firstName': row['firstName'],
            'lastName': row['lastName'],
            'roleValue': row['roleValue'],
            'positionValue': row['positionValue'],
            'image': row['image'].decode('utf-8') if isinstance(row['image'], bytes) else row['image'],
            'doctorId': int(row['doctorId']),
            'nameSpecialty': row['nameSpecialty'],
            # 'specialtyId': row['specialtyId'],    
            # 'priceId': row['priceId']
        }
        

    for i in range(1, number + 1 ):
        result.append(getName(sortedSimilarity[i][0]))

    # return jsonify({'recommended_doctors': [int(x) for x in result]})
    return jsonify({'recommended_doctors': [x for x in result]})



@app.route('/api/recommend-doctors', methods=['GET'])
def recommend_doctors():
    patient_id = request.args.get('id')
    print(patient_id)
    if patient_id is None:
        return jsonify({'error': 'Patient ID is missing'}), 400

    try:
        patient_id = int(patient_id)
    except ValueError:
        return jsonify({'error': 'Invalid Patient ID'}), 400

    # Truy vấn lịch sử khám bệnh
    query_history = f"""
    SELECT patientId, doctorId, reasonExamination
    FROM bookings
    WHERE patientId = {patient_id} AND statusId = 'S3'
    """
    history, history_columns = execute_read_query(connection, query_history)
    history_df = pd.DataFrame(history, columns=history_columns)
    print(history_df)
    if history_df.empty:
        return jsonify({'error': 'No history found for this patient'}), 400

    # Xây dựng hồ sơ bệnh nhân dựa trên các bác sĩ đã khám
    # patient_doctors = history_df['doctorId'].values
    patient_doctors = list(set(history_df['doctorId'].values))
    # print(patient_doctors)
    patient_profile = np.asarray(tfMatrix[users_df[users_df['doctorId'].isin(patient_doctors)].index].mean(axis=0))

    # print(patient_profile)

    # Tính toán độ tương đồng cosine giữa hồ sơ bệnh nhân và hồ sơ các bác sĩ
    similarity_scores = cosine_similarity(patient_profile, tfMatrix)

    # Sắp xếp các bác sĩ theo độ tương đồng
    sorted_indices = similarity_scores.argsort()[0][::-1]

    top_doctors = sorted_indices[:5]
    # print(top_doctors)

    result = []
    for index in top_doctors:
        row = users_df.iloc[index]
        result.append({
            'description': row['description'],
            'firstName': row['firstName'],
            'lastName': row['lastName'],
            'roleValue': row['roleValue'],
            'positionValue': row['positionValue'],
            'image': row['image'].decode('utf-8') if isinstance(row['image'], bytes) else row['image'],
            'doctorId': int(row['doctorId']),
            'nameSpecialty': row['nameSpecialty']
        })

    return jsonify({'recommended_doctors': result})

if __name__ == '__main__':
    app.run(port=6969)









