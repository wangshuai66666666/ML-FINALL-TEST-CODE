import time
import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def main():
    start_time = time.time()

    # --- 修改开始 ---
    # 加载数据集
    print("加载数据集...")
    # 1. 加载两个原始数据集用于训练
    data1 = pd.read_csv(r'D:\ML Final Test\output\modified_数据集Time_Series662_detail.dat')
    data2 = pd.read_csv(r'D:\ML Final Test\output\modified_数据集Time_Series661_detail.dat')

    # 2. 合并数据集以解决数据分布偏移问题，并增加训练数据量
    print("合并数据集以用于模型训练...")
    full_dataSet = pd.concat([data1, data2], ignore_index=True)
    print(f"用于训练的合并数据集形状: {full_dataSet.shape}")
    # --- 修改结束 ---

    columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                     'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
    noisy_columns = [f"{col}_noisy" for col in columns]

    # --- 修改开始 ---
    # 3. 在合并后的数据集上生成带噪声的特征，用于训练
    for col, err_col, noisy_col in zip(columns, noise_columns, noisy_columns):
        full_dataSet[noisy_col] = full_dataSet[col] + full_dataSet[err_col]

    # 4. 从合并后的数据集中定义特征和目标变量
    X_full = full_dataSet[noisy_columns].values
    y_full = full_dataSet[columns].values

    # 5. 从合并后的数据集中划分训练集和验证集，用于模型训练和早停
    print("从合并后的数据集中划分训练集和验证集...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=42  # 使用10%的数据作为验证集
    )

    print(f"训练集X的形状: {X_train.shape}")
    print(f"验证集X的形状: {X_val.shape}")
    # --- 修改结束 ---

    print("-" * 30)

    # 模型保存路径前缀
    model_path_prefix = 'catboost_denoise_model_'

    # 检查所有模型文件是否存在
    all_models_exist = all(os.path.exists(f"{model_path_prefix}{col}.cbm") for col in columns)

    # 根据文件是否存在决定流程
    if not all_models_exist:
        print("未检测到所有已保存的模型文件，将自动开始训练...")
        user_choice = '1'  # 自动选择训练
    else:
        # 文件存在时询问用户选择
        while True:
            try:
                user_choice = input("检测到已保存的模型文件，請选择操作：1-重新训练模型并保存 2-加载已有模型直接预测 (输入1或2)：")
                if user_choice in ['1', '2']:
                    break
                else:
                    print("输入无效，请重新输入1或2")
            except EOFError:
                user_choice = '1'
                print("非交互式环境，默认选择重新训练")
                break

    models = {}

    if user_choice == '1':
        print("\n开始为每个目标变量训练独立的 CatBoost 模型...")
        # 为6个目标分别训练模型
        for i, target_col in enumerate(columns):
            print(f"\n--- 正在训练第 {i + 1}/6 个模型，目标变量: {target_col} ---")

            # 获取当前目标
            y_train_single = y_train[:, i]
            y_val_single = y_val[:, i]

            # 初始化 CatBoost 回归器 (单目标)
            # 使用MAE作为损失和评估指标，直接优化我们的目标
            model = CatBoostRegressor(
                loss_function='MAE',
                eval_metric='MAE',
                iterations=2000,  # 足够的迭代次数，配合早停
                learning_rate=0.05,  # 稍微降低学习率以获得更精细的模型
                depth=6,
                l2_leaf_reg=3,
                od_type='Iter',
                od_wait=50,  # 早停耐心值
                task_type='CPU',
                verbose=200,
                random_seed=42
            )

            # 训练模型
            model.fit(
                X_train, y_train_single,
                eval_set=(X_val, y_val_single),
                use_best_model=True
            )

            # 保存模型
            model_save_path = f"{model_path_prefix}{target_col}.cbm"
            model.save_model(model_save_path)
            print(f"模型 {target_col} 已保存至 {model_save_path}")

            # 将训练好的模型存入字典
            models[target_col] = model

        print("\n所有模型训练完成！")
    else:
        try:
            print(f"\n开始加载所有已保存的模型...")
            for target_col in columns:
                model_load_path = f"{model_path_prefix}{target_col}.cbm"
                model = CatBoostRegressor()
                model.load_model(model_load_path)
                models[target_col] = model
                print(f"模型 {target_col} 加载成功。")
            print("所有模型加载完成。")
        except (FileNotFoundError, Exception) as e:
            print(f"错误：{str(e)}，将自动重新训练...")
            # 若加载失败，自动重新训练（复制上面的训练逻辑）
            for i, target_col in enumerate(columns):
                print(f"\n--- 正在重新训练第 {i + 1}/6 个模型，目标变量: {target_col} ---")
                y_train_single = y_train[:, i]
                y_val_single = y_val[:, i]
                model = CatBoostRegressor(loss_function='MAE', eval_metric='MAE', iterations=2000, learning_rate=0.05,
                                          depth=6, l2_leaf_reg=3, od_type='Iter', od_wait=50, task_type='CPU',
                                          verbose=200, random_seed=42)
                model.fit(X_train, y_train_single, eval_set=(X_val, y_val_single), use_best_model=True)
                model_save_path = f"{model_path_prefix}{target_col}.cbm"
                model.save_model(model_save_path)
                models[target_col] = model
            print("所有模型已重新训练并保存。")

    # --- 修改开始 ---
    # 预测阶段：加载完整的661数据集进行预测和评估
    print("\n模型训练/加载完成，开始对数据集661进行完整预测...")
    # 重新加载data1用于预测，确保与评估脚本一致
    prediction_data = pd.read_csv(r'D:\ML Final Test\dataset\数据集（含真实值）\modified_数据集Time_Series662.dat')
    print(f"加载了 {len(prediction_data)} 条数据用于预测。")

    # 为prediction_data创建带噪声的特征
    for col, err_col, noisy_col in zip(columns, noise_columns, noisy_columns):
        prediction_data[noisy_col] = prediction_data[col] + prediction_data[err_col]

    # 定义用于预测的特征和用于评估的真实值
    X_to_predict = prediction_data[noisy_columns].values
    y_true_for_eval = prediction_data[columns].values

    # 使用所有模型进行预测
    y_pred_list = []
    for target_col in columns:
        model = models[target_col]
        pred = model.predict(X_to_predict)
        y_pred_list.append(pred)

    # 将预测结果列表转换为二维数组
    y_pred = np.column_stack(y_pred_list)
    # --- 修改结束 ---

    # 结果评估与保存 - 格式兼容evaluate_predictions.py
    results = []
    for true_val, pred_val in zip(y_true_for_eval, y_pred):
        error = np.abs(true_val - pred_val)
        results.append([
            ' '.join(map(lambda x: f"{x:.4f}", true_val)),
            ' '.join(map(lambda x: f"{x:.4f}", pred_val)),
            ' '.join(map(lambda x: f"{x:.4f}", error))
        ])

    # 保存结果到comparison目录
    output_path = r'D:\ML Final Test\comparison\catboost_multi_model_result.csv'
    result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存到 {output_path}")

    # 同时保存一份到当前目录
    result_df.to_csv("catboost_multi_model_result.csv", index=False)
    print(f"预测结果已保存到 catboost_multi_model_result.csv")

    # 计算并打印MAE
    error_values = result_df['Error'].str.split(' ', expand=True).astype(float).values
    mae_per_feature = np.mean(error_values, axis=0)

    print("\n6个输出特征的平均绝对误差（MAE）为：")
    for i, col in enumerate(columns):
        print(f"{col}: {mae_per_feature[i]:.4f}")

    total_mae = np.mean(mae_per_feature)
    print("-" * 30)
    print(f"总平均绝对误差（MAE）: {total_mae:.4f}")

    end_time = time.time()
    print(f"总耗时：{end_time - start_time:.3f}秒")


if __name__ == "__main__":
    main()
