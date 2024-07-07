import mysql.connector
from mysql.connector import Error
import sys
import pandas as pd
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
SELECT doctor_infors.*, users.image, users.firstName, users.lastName, description_alls.description, role_codes.valueVI AS roleValue, position_codes.valueVI AS positionValue
FROM doctor_infors
JOIN users ON doctor_infors.doctorId = users.id
JOIN description_alls ON doctor_infors.doctorId = description_alls.doctorId
JOIN allcodes AS role_codes ON users.roleId = role_codes.keyMap
JOIN allcodes AS position_codes ON users.positionId = position_codes.keyMap

"""
users, columns = execute_read_query(connection, select_users)
# JOIN schedules ON doctor_infors.doctorId = schedules.doctorId
# GROUP BY doctor_infors.doctorId, schedules.date
# schedules.date, schedules.timeType
# ,  schedules.date, GROUP_CONCAT(schedules.timeType SEPARATOR ' ') AS timeTypes

# Chuyển đổi users thành DataFrame
users_df = pd.DataFrame(users, columns=columns)
# print(users_df)


# # Gộp các khung giờ lại thành một chuỗi duy nhất
# users_df['timeType'] = users_df['timeType'].astype(str)  # Đảm bảo các giá trị là chuỗi
# users_df['combineDateTime'] = users_df.groupby(['doctorId', 'date'])['timeType'].transform(lambda x: ' '.join(x))

# # Loại bỏ các hàng trùng lặp
# users_df = users_df.drop_duplicates(subset=['doctorId', 'date', 'combineDateTime'])
# users_df = users_df.drop(columns=['date', 'timeType'])


# Thêm cột combineFeatures
def combine_features(row):
    return f"{row['provinceId']} {row['specialtyId']} {row['priceId']} {row['healthFacilitiesId']} {row['description']}  "

# {row['date']} {row['timeType']} 
# {row['combineDateTime']}
users_df['combineFeatures'] = users_df.apply(combine_features, axis=1)
print(users_df)
tf = TfidfVectorizer()
tfMatrix = tf.fit_transform(users_df['combineFeatures'])

# {row['provinceId']} {row['specialtyId']} {row['priceId']} {row['healthFacilitiesId']} {row['description']}

sililar = cosine_similarity(tfMatrix)

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
            # 'provinceId': row['provinceId'],
            # 'specialtyId': row['specialtyId'],
            # 'priceId': row['priceId']
        }
        

    for i in range(1, number + 1):
        result.append(getName(sortedSimilarity[i][0]))

    # return jsonify({'recommended_doctors': [int(x) for x in result]})
    return jsonify({'recommended_doctors': [x for x in result]})

if __name__ == '__main__':
    app.run(port=6969)
