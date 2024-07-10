from data_loader import load_data
from model_builder import build_model
from train_and_evaluate import train_and_evaluate

#設定項目
start_epoch = 20
patience = 5
fit_type = "bfloat16"
numpy_save_folder = './numpy_data/1/'
input_shape = (100, 120, 1)

# 数据加载和预处理
train_data = load_data(numpy_save_folder,fit_type=fit_type)

# 模型构建
model = build_model(input_shape, train_data[4])

# 训练和评估
train_and_evaluate(
    model = model,
    train_data = train_data, 
    fit_type = fit_type, 
    batch_size = 128,
    start_epoch = start_epoch,
    patience = patience
)
