import time
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error
from joblib import dump, load

# 尝试导入CatBoost和XGBoost
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("警告: CatBoost库未安装，将无法使用CatBoost模型")
    CatBoostRegressor = None

try:
    import xgboost as xgb
except ImportError:
    print("警告: XGBoost库未安装，将无法使用XGBoost模型")
    xgb = None


def add_statistical_features(data, columns, window_sizes=[5, 10, 20]):
    """基于输入列（此处为noise_columns）添加统计特征"""
    new_features = []
    new_columns_dict = {}

    for col in columns:
        # 滚动窗口统计（反映时序趋势和波动）
        for window in window_sizes:
            new_columns_dict[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window, min_periods=1).mean()
            new_columns_dict[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window, min_periods=1).std()
            new_columns_dict[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window, min_periods=1).min()
            new_columns_dict[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window, min_periods=1).max()
            new_features.extend([
                f'{col}_rolling_mean_{window}',
                f'{col}_rolling_std_{window}',
                f'{col}_rolling_min_{window}',
                f'{col}_rolling_max_{window}'
            ])

        # 差分特征（反映变化率）
        new_columns_dict[f'{col}_diff_1'] = data[col].diff(1).fillna(0)
        new_columns_dict[f'{col}_diff_2'] = data[col].diff(2).fillna(0)
        new_features.extend([f'{col}_diff_1', f'{col}_diff_2'])

        # 转换特征（增强非线性表达）
        new_columns_dict[f'{col}_abs'] = data[col].abs()
        new_columns_dict[f'{col}_square'] = data[col] ** 2
        new_features.extend([f'{col}_abs', f'{col}_square'])

    new_features_df = pd.DataFrame(new_columns_dict)
    data = pd.concat([data, new_features_df], axis=1)
    return data, new_features


def add_interaction_features(data, columns, max_interactions=5):
    """基于输入列（此处为noise_columns）添加特征交互"""
    new_features = []
    new_columns_dict = {}
    count = 0

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if count >= max_interactions:
                break

            col1 = columns[i]
            col2 = columns[j]

            new_columns_dict[f'{col1}_times_{col2}'] = data[col1] * data[col2]
            new_columns_dict[f'{col1}_div_{col2}'] = data[col1] / (data[col2] + 1e-8)  # 避免除零

            new_features.extend([
                f'{col1}_times_{col2}',
                f'{col1}_div_{col2}'
            ])
            count += 1
        if count >= max_interactions:
            break

    new_features_df = pd.DataFrame(new_columns_dict)
    data = pd.concat([data, new_features_df], axis=1)
    return data, new_features


def get_optimized_lgb_params():
    """LightGBM参数（将verbose移到模型初始化）"""
    return {
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.2,
        'reg_lambda': 0.3,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 100  # 控制训练日志输出频率（每100轮输出一次）
    }


def get_optimized_catboost_params():
    """CatBoost参数"""
    return {
        'iterations': 2000,
        'depth': 8,
        'learning_rate': 0.005,
        'l2_leaf_reg': 3,
        'border_count': 64,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',  # CatBoost在模型参数中指定评估指标
        'random_seed': 42,
        'verbose': 50  # 每50轮输出一次日志
    }


def get_optimized_xgb_params():
    """XGBoost参数（将eval_metric移到模型初始化）"""
    return {
        'n_estimators': 2000,
        'max_depth': 10,
        'learning_rate': 0.005,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',  # XGBoost在模型参数中指定评估指标
        'random_state': 42,
        'verbosity': 1  # 控制日志输出（1=警告，2=信息）
    }


def handle_outliers(X, method='iqr', threshold=3, window_size=50):
    """处理异常值"""
    X_out = X.copy()
    if method == 'zscore':
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        mask = z_scores < threshold
        for col in range(X.shape[1]):
            X_out[:, col][~mask[:, col]] = X[:, col].mean()
    elif method == 'iqr':
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        X_out = np.clip(X, Q1 - threshold * IQR, Q3 + threshold * IQR)
    elif method == 'rolling':
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in X_df.columns:
            rolling_mean = X_df[col].rolling(window=window_size, min_periods=1).mean()
            rolling_std = X_df[col].rolling(window=window_size, min_periods=1).std().replace(0, 1)
            z_scores = np.abs((X_df[col] - rolling_mean) / rolling_std)
            outliers = z_scores > threshold
            if outliers.any():
                col_data = X_df[col].copy()
                col_data[outliers] = col_data.shift(1)[outliers].fillna(rolling_mean[outliers])
                X_df[col] = col_data
        X_out = X_df.values if not isinstance(X, pd.DataFrame) else X_df
    return X_out


def create_preprocessor(scaler_type='robust'):
    """创建特征缩放器"""
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    elif scaler_type == 'robust':
        return RobustScaler()
    elif scaler_type == 'quantile':
        return QuantileTransformer(output_distribution='normal', random_state=42)
    else:
        raise ValueError(f"不支持的缩放器类型: {scaler_type}")


def train_single_model(model_type, X_train, y_train, X_val, y_val, feature_names):
    """训练单个模型（修复参数位置兼容问题）"""
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)

    if model_type == 'lgb':
        # LightGBM：verbose在模型参数中，fit中无需重复传递
        model = lgb.LGBMRegressor(**get_optimized_lgb_params())
        model.fit(
            X_train_df, y_train,
            eval_set=[(X_val_df, y_val)],  # 验证集
        )
    elif model_type == 'catboost' and CatBoostRegressor is not None:
        # CatBoost：eval_metric在模型参数中，fit中传递验证集
        model = CatBoostRegressor(**get_optimized_catboost_params())
        model.fit(
            X_train_df, y_train,
            eval_set=(X_val_df, y_val),  # 验证集
            use_best_model=True  # 使用验证集最优模型
        )
    elif model_type == 'xgb' and xgb is not None:
        # XGBoost：eval_metric在模型参数中，fit中无需传递
        model = xgb.XGBRegressor(**get_optimized_xgb_params())
        model.fit(
            X_train_df, y_train,
            eval_set=[(X_val_df, y_val)],  # 验证集
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    y_pred_val = model.predict(X_val_df)
    return model, mean_absolute_error(y_val, y_pred_val)


def train_ensemble_models(X_train, y_train, X_test, y_test, target_columns, feature_names, preprocess_config=None):
    """训练集成模型"""
    model_types = ['lgb']
    if CatBoostRegressor is not None:
        model_types.append('catboost')
    if xgb is not None:
        model_types.append('xgb')

    models = {}
    scalers = {}
    model_weights = {}
    target_scalers = {}

    # 确保配置完整（含normalize_target）
    preprocess_config = preprocess_config or {
        'scaler_type': 'robust',
        'handle_outliers': True,
        'outlier_method': 'rolling',
        'val_split_ratio': 0.2,
        'normalize_target': False  # 树模型无需目标标准化
    }

    # 预处理X（基于noise_columns及其特征）
    X_train_processed = handle_outliers(X_train, method=preprocess_config['outlier_method']) if preprocess_config[
        'handle_outliers'] else X_train
    X_test_processed = handle_outliers(X_test, method=preprocess_config['outlier_method']) if preprocess_config[
        'handle_outliers'] else X_test

    for i, target_col in enumerate(target_columns):
        print(f"\n训练目标特征: {target_col}")
        y_train_single = y_train[target_col].values
        y_test_single = y_test[target_col].values

        # 划分验证集
        val_size = int(len(X_train_processed) * preprocess_config['val_split_ratio'])
        X_val, y_val = X_train_processed[-val_size:], y_train_single[-val_size:]
        X_train_subset, y_train_subset = X_train_processed[:-val_size], y_train_single[:-val_size]

        # 特征缩放
        scaler = create_preprocessor(preprocess_config['scaler_type'])
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_processed)

        # 目标值标准化（按配置执行，默认关闭）
        if preprocess_config['normalize_target']:
            target_scaler = RobustScaler()
            y_train_scaled = target_scaler.fit_transform(y_train_subset.reshape(-1, 1)).flatten()
            y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            target_scalers[target_col] = target_scaler
        else:
            y_train_scaled, y_val_scaled = y_train_subset, y_val

        # 训练各模型并计算权重
        val_mae_dict = {}
        final_models = {}
        for model_type in model_types:
            print(f"  训练{model_type.upper()}...")
            model, val_mae = train_single_model(
                model_type, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, feature_names
            )
            final_models[model_type] = model
            val_mae_dict[model_type] = val_mae
            print(f"  {model_type.upper()}验证MAE: {val_mae:.4f}")

        # 计算模型权重（MAE越小，权重越高）
        total_mae_inverse = sum(1 / mae for mae in val_mae_dict.values())
        weights = {k: (1 / v) / total_mae_inverse for k, v in val_mae_dict.items()}

        models[target_col] = final_models
        scalers[target_col] = scaler
        model_weights[target_col] = weights

    return models, scalers, model_weights, target_scalers


def ensemble_predict(models, scalers, model_weights, X_test, feature_names, target_scalers=None):
    """集成预测"""
    predictions = {}
    for target_col, model_dict in models.items():
        scaler = scalers[target_col]
        weights = model_weights[target_col]
        X_test_scaled = scaler.transform(X_test)
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

        weighted_pred = np.zeros(X_test.shape[0])
        for model_type, model in model_dict.items():
            weighted_pred += model.predict(X_test_df) * weights[model_type]

        # 反标准化目标值（如果需要）
        if target_scalers and target_col in target_scalers:
            weighted_pred = target_scalers[target_col].inverse_transform(weighted_pred.reshape(-1, 1)).flatten()
        predictions[target_col] = weighted_pred
    return predictions


def main():
    start_time = time.time()

    # 1. 定义列名（按要求：columns是原始列，noise_columns是噪声列）
    columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                     'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

    # 2. 加载数据（请替换为你的实际路径）
    print("加载数据集...")
    train_dataSet = pd.read_csv(r'D:\ML Final Test\output\modified_数据集Time_Series662_detail.dat')
    test_dataSet = pd.read_csv(r'D:\ML Final Test\output\modified_数据集Time_Series661_detail.dat')

    # 3. 特征工程（基于noise_columns生成统计/交互特征）
    print("基于噪声列生成特征...")
    # 3.1 统计特征（滚动窗口、差分等）
    train_dataSet, stat_features = add_statistical_features(train_dataSet, noise_columns)
    test_dataSet, _ = add_statistical_features(test_dataSet, noise_columns)  # 测试集同步处理
    # 3.2 交互特征（噪声列之间的乘积/除法）
    train_dataSet, interaction_features = add_interaction_features(train_dataSet, noise_columns)
    test_dataSet, _ = add_interaction_features(test_dataSet, noise_columns)  # 测试集同步处理

    # 4. 定义X的特征集（原始噪声列 + 生成的特征）
    all_features = noise_columns + stat_features + interaction_features

    # 5. 处理缺失值
    train_dataSet = train_dataSet.fillna(0)
    test_dataSet = test_dataSet.fillna(0)

    # 6. 划分X和y（核心：X用噪声列及其特征，y用原始列）
    X_train = train_dataSet[all_features].values  # X：噪声列+生成特征
    y_train = train_dataSet[columns]  # y：原始列
    X_test = test_dataSet[all_features].values  # X：噪声列+生成特征
    y_test = test_dataSet[columns]  # y：原始列

    print(f"特征集大小: {len(all_features)} | X_train形状: {X_train.shape} | y_train形状: {y_train.shape}")

    # 7. 模型训练/加载
    model_dir = 'ensemble_models'
    os.makedirs(model_dir, exist_ok=True)
    models_exist = True

    # 检查模型是否存在
    try:
        for target_col in columns:
            if not os.path.exists(os.path.join(model_dir, f'model_weights_{target_col}.joblib')):
                models_exist = False
                break
    except:
        models_exist = False

    if not models_exist:
        print("训练新模型...")
        preprocess_config = {
            'scaler_type': 'robust',
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'val_split_ratio': 0.2,
            'normalize_target': False  # 树模型无需目标标准化
        }
        models, scalers, model_weights, target_scalers = train_ensemble_models(
            X_train, y_train, X_test, y_test, columns, all_features, preprocess_config
        )
        # 保存模型
        for target_col in columns:
            dump(models[target_col], os.path.join(model_dir, f'models_{target_col}.joblib'))
            dump(scalers[target_col], os.path.join(model_dir, f'scaler_{target_col}.joblib'))
            dump(model_weights[target_col], os.path.join(model_dir, f'model_weights_{target_col}.joblib'))
        if target_scalers:
            dump(target_scalers, os.path.join(model_dir, 'target_scalers.joblib'))
    else:
        print("加载已有模型...")
        models = {col: load(os.path.join(model_dir, f'models_{col}.joblib')) for col in columns}
        scalers = {col: load(os.path.join(model_dir, f'scaler_{col}.joblib')) for col in columns}
        model_weights = {col: load(os.path.join(model_dir, f'model_weights_{col}.joblib')) for col in columns}
        target_scalers = load(os.path.join(model_dir, 'target_scalers.joblib')) if os.path.exists(
            os.path.join(model_dir, 'target_scalers.joblib')) else None

    # 8. 预测与评估
    print("\n预测中...")
    predictions = ensemble_predict(models, scalers, model_weights, X_test, all_features, target_scalers)

    # 计算MAE
    overall_mae = []
    for col in columns:
        mae = mean_absolute_error(y_test[col], predictions[col])
        overall_mae.append(mae)
        print(f"{col} 的测试MAE: {mae:.4f}")
    print(f"平均MAE: {np.mean(overall_mae):.4f}")

    # 保存结果
    results_dir = 'prediction_results'
    os.makedirs(results_dir, exist_ok=True)
    result_df = pd.DataFrame({
        **{f'True_{col}': y_test[col].values for col in columns},
        **{f'Pred_{col}': predictions[col] for col in columns}
    })
    result_df.to_csv(os.path.join(results_dir, 'ensemble_results.csv'), index=False)
    print(f"结果已保存至 {results_dir}")

    print(f"\n总耗时: {time.time() - start_time:.2f}秒")


if __name__ == "__main__":
    main()