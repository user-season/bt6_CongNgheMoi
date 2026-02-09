import joblib
import os
import numpy as np
from django.conf import settings

class IrisModelLoader:
    _model = None

    @classmethod
    def load(cls):
        if cls._model is None:
            # Đường dẫn tới file model
            parent_dir = os.path.dirname(settings.BASE_DIR)
            save_dir = os.path.join(parent_dir, 'train_pipeline', 'saved_models')
            model_path = os.path.join(save_dir, 'iris_model.pkl')
            
            if os.path.exists(model_path):
                cls._model = joblib.load(model_path)
        return cls._model

    @classmethod
    def predict(cls, features):
        """
        Trả về: (Tên lớp dự đoán, Độ tin cậy cao nhất, Dictionary xác suất chi tiết)
        """
        model = cls.load()
        if model is None:
            return None, 0.0, {}

        # features: [sepal_l, sepal_w, petal_l, petal_w]
        input_data = [features]
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        
        try:
            # Lấy danh sách xác suất [p1, p2, p3]
            probas = model.predict_proba(input_data)[0] 
            
            # 1. Tìm xác suất lớn nhất
            max_prob = np.max(probas) * 100
            prediction_idx = np.argmax(probas)
            predicted_class = class_names[prediction_idx]
            
            # 2. Tạo dictionary chi tiết: {'Setosa': 10.5, 'Versicolor': 80.0, ...}
            detailed_proba = {
                name: round(prob * 100, 2) 
                for name, prob in zip(class_names, probas)
            }

        except AttributeError:
            # Fallback nếu model không hỗ trợ predict_proba
            prediction_idx = model.predict(input_data)[0]
            predicted_class = class_names[prediction_idx]
            max_prob = 100.0
            detailed_proba = {predicted_class: 100.0}

        return predicted_class, max_prob, detailed_proba