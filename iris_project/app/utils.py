import os
import pandas as pd
from django.conf import settings
    


def get_global_limits():
    """Hàm lấy giới hạn Min/Max của toàn bộ dataset để validate đầu vào"""
    try:
        parent_dir = os.path.dirname(settings.BASE_DIR)
        data_path = os.path.join(parent_dir, 'train_pipeline', 'data', 'iris.csv')
        
        if not os.path.exists(data_path):
            print('read file data')
            return None

        df = pd.read_csv(data_path)
        
        # Tạo dictionary chứa min/max của từng cột
        # Lưu ý: Tên cột phải khớp với file CSV của bạn (SepalLengthCm, v.v.)
        limits = {
            'sepal_length': {'min': float(df['SepalLengthCm'].min()), 'max': float(df['SepalLengthCm'].max())},
            'sepal_width':  {'min': float(df['SepalWidthCm'].min()),  'max': float(df['SepalWidthCm'].max())},
            'petal_length': {'min': float(df['PetalLengthCm'].min()), 'max': float(df['PetalLengthCm'].max())},
            'petal_width':  {'min': float(df['PetalWidthCm'].min()),  'max': float(df['PetalWidthCm'].max())},
        }
        return limits

    except Exception as e:
        print(f"Lỗi lấy limits: {e}")
        return None