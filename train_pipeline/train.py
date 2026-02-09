import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from django.conf import settings

# --- HÀM MỚI: Tính toán thống kê Min/Max từ CSV ---
def get_dataset_stats():
    try:
        parent_dir = os.path.dirname(settings.BASE_DIR)
        data_path = os.path.join(parent_dir, 'train_pipeline', 'data', 'Iris.csv')
        
        if not os.path.exists(data_path):
            return {}

        df = pd.read_csv(data_path)
        
        # Nhóm theo loài hoa (Species) và tìm Min, Max
        # Giả sử tên cột trong CSV là: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
        # Nếu CSV của bạn tên cột khác (ví dụ sepal_length), hãy sửa lại ở đây
        stats = df.groupby('Species').agg({
            'SepalLengthCm': ['min', 'max'],
            'SepalWidthCm': ['min', 'max'],
            'PetalLengthCm': ['min', 'max'],
            'PetalWidthCm': ['min', 'max']
        })

        # Chuyển đổi sang format dễ dùng trong Django Template
        formatted_stats = {}
        colors = {'Iris-setosa': 'text-primary', 'Iris-versicolor': 'text-success', 'Iris-virginica': 'text-danger'}
        
        for species in stats.index:
            # Xử lý tên hiển thị (bỏ chữ Iris- nếu muốn gọn)
            display_name = species.replace('Iris-', '').capitalize()
            
            formatted_stats[species] = {
                'name': display_name,
                'color_class': colors.get(species, 'text-dark'), # Mặc định đen nếu không khớp
                'sl_min': stats.loc[species, ('SepalLengthCm', 'min')],
                'sl_max': stats.loc[species, ('SepalLengthCm', 'max')],
                'sw_min': stats.loc[species, ('SepalWidthCm', 'min')],
                'sw_max': stats.loc[species, ('SepalWidthCm', 'max')],
                'pl_min': stats.loc[species, ('PetalLengthCm', 'min')],
                'pl_max': stats.loc[species, ('PetalLengthCm', 'max')],
                'pw_min': stats.loc[species, ('PetalWidthCm', 'min')],
                'pw_max': stats.loc[species, ('PetalWidthCm', 'max')],
            }
            
        return formatted_stats
    except Exception as e:
        print(f"Lỗi tính stats: {e}")
        return {}

# --- HÀM CŨ (Đã cập nhật để trả về cả stats) ---
def run_training():
    try:
        parent_dir = os.path.dirname(settings.BASE_DIR)
        data_path = os.path.join(parent_dir, 'train_pipeline', 'data', 'Iris.csv')

        if not os.path.exists(data_path):
            return f"Lỗi: Không tìm thấy file tại {data_path}"

        df = pd.read_csv(data_path)
        
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
            
        X = df.drop('Species', axis=1)
        y = df['Species']
        
        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        def get_acc(X_data, y_data):
            if len(X_data) > 0:
                return accuracy_score(y_data, model.predict(X_data))
            return 0.0

        metrics = {
            "train_acc": get_acc(X_train, y_train),
            "val_acc": get_acc(X_val, y_val),
            "test_acc": get_acc(X_test, y_test)
        }

        save_dir = os.path.join(parent_dir, 'train_pipeline', 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'iris_model.pkl')
        joblib.dump(model, model_path)
        
        return metrics

    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}"