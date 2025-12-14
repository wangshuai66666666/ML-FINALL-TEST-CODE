import time
import os  # 用于检查文件是否存在
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'D:\ML Final Test\output\modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'D:\ML Final Test\output\modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 生成带噪声的特征
noisy_columns = [f"{col}_noisy" for col in columns]
for col, err_col, noisy_col in zip(columns, noise_columns, noisy_columns):
    train_dataSet[noisy_col] = train_dataSet[col] + train_dataSet[err_col]
    test_dataSet[noisy_col] = test_dataSet[col] + test_dataSet[err_col]

# 定义特征和目标变量
X_train = train_dataSet[noisy_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noisy_columns]
y_test = test_dataSet[columns]

print("数据预处理完成。")
print(f"训练集X的形状: {X_train.shape}")
print(f"训练集y的形状: {y_train.shape}")
print("-" * 30)

"""模型定义"""
model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    alpha=0.001,
    verbose=True,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    batch_size=64,
)

# 模型和标准化器保存路径
model_path = 'mlp_model.joblib'
scaler_path = 'scaler.joblib'

# 检查参数文件是否存在
model_exists = os.path.exists(model_path)
scaler_exists = os.path.exists(scaler_path)
files_exist = model_exists and scaler_exists  # 只有两个文件都存在时才视为完整

# 根据文件是否存在决定流程
if not files_exist:
    print("未检测到已保存的模型和标准化器文件，将自动开始训练...")
    user_choice = '1'  # 自动选择训练
else:
    # 文件存在时询问用户选择
    while True:
        user_choice = input("检测到已保存的模型文件，請选择操作：1-重新训练模型并保存 2-加载已有模型直接预测 (输入1或2)：")
        if user_choice in ['1', '2']:
            break
        else:
            print("输入无效，请重新输入1或2")

# 根据选择执行对应操作
if user_choice == '1':
    print("\n开始特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("开始训练模型...")
    model.fit(X_train_scaled, y_train)
    print("模型训练完成。")

    # 保存模型和标准化器
    dump(model, model_path)
    dump(scaler, scaler_path)
    print(f"模型已保存至 {model_path}")
    print(f"特征标准化器已保存至 {scaler_path}")
else:
    try:
        print(f"\n开始加载模型 {model_path} 和标准化器 {scaler_path}...")
        model = load(model_path)
        scaler = load(scaler_path)
        X_test_scaled = scaler.transform(X_test)
        print("模型和标准化器加载完成。")
    except FileNotFoundError as e:
        print(f"错误：未找到文件 {e.filename}，将自动重新训练...")
        # 若加载失败，自动重新训练
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        dump(model, model_path)
        dump(scaler, scaler_path)
        print(f"模型和标准化器已重新训练并保存至 {model_path} 和 {scaler_path}")

# 预测
y_predict = model.predict(X_test_scaled)

# 结果评估与保存
results = []
for true_val, pred_val in zip(y_test.values, y_predict):
    error = np.abs(true_val - pred_val)
    results.append([
        ' '.join(map(str, true_val)),
        ' '.join(map(str, pred_val)),
        ' '.join(map(str, error))
    ])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP.csv", index=False)
print("\n预测结果已保存到 result_MLP.csv")

print("-" * 50)

# 计算平均绝对误差
data = pd.read_csv("result_MLP.csv")
error_columns = data['Error'].str.split(' ', expand=True).apply(pd.to_numeric)
mae_per_feature = error_columns.mean()

print("6个输出特征的平均绝对误差（MAE）为：")
for i, col in enumerate(columns):
    print(f"{col}: {mae_per_feature[i]:.4f}")

total_mae = mae_per_feature.mean()
print("-" * 30)
print(f"总平均绝对误差（MAE）: {total_mae:.4f}")

end_time = time.time()
print(f"总耗时：{end_time - start_time:.3f}秒")