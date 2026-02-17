from django.shortcuts import render, redirect
from .load_model import IrisModelLoader
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from train_pipeline.train import run_training, get_dataset_stats 


# Import hàm mới
from .utils import get_global_limits 

def index(request):
    result = None
    prob_msg = None
    detailed_proba = None
    error_msg = None
    input_data = {}

    # 1. LẤY GIỚI HẠN DỮ LIỆU TỪ DATASET
    global_limits = get_global_limits()
    
    # Nếu không đọc được file (ví dụ lần đầu chạy), ta set giá trị mặc định an toàn
    if not global_limits:
        global_limits = {
            'sepal_length': {'min': 0.1, 'max': 10.0},
            'sepal_width':  {'min': 0.1, 'max': 10.0},
            'petal_length': {'min': 0.1, 'max': 10.0},
            'petal_width':  {'min': 0.1, 'max': 10.0},
        }

    # Lấy thống kê chi tiết cho bảng tham chiếu (code cũ)
    stats_data = get_dataset_stats()

    if request.method == 'POST' and 'predict_btn' in request.POST:
        try:
            # Lấy dữ liệu từ form
            sl_raw = request.POST.get('sepal_length')
            sw_raw = request.POST.get('sepal_width')
            pl_raw = request.POST.get('petal_length')
            pw_raw = request.POST.get('petal_width')
            
            input_data = {'sl': sl_raw, 'sw': sw_raw, 'pl': pl_raw, 'pw': pw_raw}

            # Ép kiểu float
            sl, sw, pl, pw = float(sl_raw), float(sw_raw), float(pl_raw), float(pw_raw)

            # 2. KIỂM TRA LOGIC DỰA TRÊN DATASET (Có nới lỏng biên độ 1 xíu để linh hoạt)
            # Ví dụ: Cho phép nhập lớn hơn Max của dataset một chút (buffer)
            BUFFER = 0 # Cho phép sai số
            
            # Hàm kiểm tra nhanh
            def check_limit(val, name, key):
                limit = global_limits[key]
                if val > limit['max'] + BUFFER:
                    raise ValueError(f"{name} quá lớn! (Dữ liệu thực tế Max chỉ {limit['max']}cm)")
                elif val < limit['min'] + BUFFER:
                    raise ValueError(f"{name} quá bé! (Dữ liệu thực tế Min chỉ {limit['min']}cm)")

            check_limit(sl, "Chiều dài Đài hoa", 'sepal_length')
            check_limit(sw, "Chiều rộng Đài hoa", 'sepal_width')
            check_limit(pl, "Chiều dài Cánh hoa", 'petal_length')
            check_limit(pw, "Chiều rộng Cánh hoa", 'petal_width')

            # --- DỰ ĐOÁN ---
            pred_name, confidence, detailed_proba = IrisModelLoader.predict([sl, sw, pl, pw])
            
            if pred_name: 
                # if confidence < 60:
            #         result = "Không xác định"
            #         prob_msg = "Chỉ số không khớp với bất kỳ dữ liệu nào đã học."
            #     else:
                result = pred_name
                prob_msg = f"Độ chính xác: {confidence:.2f}%"

        except ValueError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"Lỗi: {str(e)}"

    return render(request, 'index.html', {
        'result': result, 
        'prob_msg': prob_msg,
        'detailed_proba': detailed_proba,
        'error_msg': error_msg,
        'input_data': input_data,
        'train_msg': request.session.pop('train_msg', None),
        'stats_data': stats_data,
        'limits': global_limits
    })



def train(request):
    if request.method == 'POST':
        if run_training is None:
             request.session['train_msg'] = "❌ Module training không tìm thấy."
             return redirect('index')

        try:
            result = run_training()
            
            # Kiểm tra kết quả
            if isinstance(result, dict):
                IrisModelLoader._model = None # Reset cache singleton để load model mới
                
                msg = (
                    f"✅ <b>Huấn luyện thành công!</b><br>"
                    f"<small>Train Acc: {result.get('train_acc', 0):.2%} | "
                    f"Test Acc: {result.get('test_acc', 0):.2%}</small>"
                )
                request.session['train_msg'] = msg
            else:
                request.session['train_msg'] = f"❌ Thất bại: {result}"
        except Exception as e:
            request.session['train_msg'] = f"❌ Lỗi khi train: {str(e)}"
        
    return redirect('index')