# 本文件用于训练DeepCFD模型
import pickle
import json
from paddle.distributed import fleet
from utils.train_functions import *
from utils.functions import *
from model.UNetEx import UNetEx
import configparser

if __name__ == "__main__":
    fleet.init(is_collective=True)
    config = configparser.ConfigParser()
    config.read("./config/config.ini")

    # 加载数据集并处理
    x = pickle.load(open(os.path.join(config["path"]["data_path"], "dataX.pkl"), "rb"))
    y = pickle.load(open(os.path.join(config["path"]["data_path"], "dataY.pkl"), "rb"))
    x = paddle.to_tensor(x, dtype="float32")
    y = paddle.to_tensor(y, dtype="float32")
    y_trans = paddle.transpose(y, perm=[0, 2, 3, 1])
    channels_weights = paddle.reshape(
        paddle.sqrt(paddle.mean(paddle.transpose(y, perm=[0, 2, 3, 1]).reshape((981 * 172 * 79, 3)) ** 2, axis=0)),
        shape=[1, -1, 1, 1])

    # 创建保存文件夹
    simulation_directory = config["path"]["save_path"]
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)

    # 按7：3的比例分割数据集，7为训练集，3为测试集
    train_data, test_data = split_tensors(x, y, ratio=float(config["hyperparameter"]["train_test_ratio"]))

    train_dataset, test_dataset = paddle.io.TensorDataset([train_data[0], train_data[1]]), \
                                  paddle.io.TensorDataset([test_data[0], test_data[1]])
    test_x, test_y = test_dataset[:]

    # 设定种子，便于复现
    paddle.seed(999)
    # 设置训练epochs和batch_size
    epochs = int(config["hyperparameter"]["epochs"])
    batch_size = int(config["hyperparameter"]["batch_size"])
    # 设置学习率
    lr = float(config["hyperparameter"]["learning_rate"])
    # 设置卷积核大小
    kernel_size = int(config["net_parameter"]["kernel_size"])
    # 设置卷积层channel数目
    filters = [int(i) for i in config["net_parameter"]["filters"].split(",")]
    # 设置batch_norm和weight_norm
    bn = bool(int(config["net_parameter"]["batch_norm"]))
    wn = bool(int(config["net_parameter"]["weight_norm"]))
    # 构建模型
    model = UNetEx(3, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    model = fleet.distributed_model(model)
    # 定义优化器
    optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(),
                                       weight_decay=float(config["hyperparameter"]["weight_decay"]))
    optimizer = fleet.distributed_optimizer(optimizer)

    # 设置记录列表
    config = {}
    train_loss_curve = []
    test_loss_curve = []
    train_mse_curve = []
    test_mse_curve = []
    train_ux_curve = []
    test_ux_curve = []
    train_uy_curve = []
    test_uy_curve = []
    train_p_curve = []
    test_p_curve = []


    # 用于后续训练过程的记录
    def after_epoch(scope):
        train_loss_curve.append(scope["train_loss"])
        test_loss_curve.append(scope["val_loss"])
        train_mse_curve.append(scope["train_metrics"]["mse"])
        test_mse_curve.append(scope["val_metrics"]["mse"])
        train_ux_curve.append(scope["train_metrics"]["ux"])
        test_ux_curve.append(scope["val_metrics"]["ux"])
        train_uy_curve.append(scope["train_metrics"]["uy"])
        test_uy_curve.append(scope["val_metrics"]["uy"])
        train_p_curve.append(scope["train_metrics"]["p"])
        test_p_curve.append(scope["val_metrics"]["p"])


    # 损失函数
    def loss_func(model, batch):
        x, y = batch
        output = model(x)
        lossu = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        lossv = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        lossp = paddle.abs((output[:, 2, :, :] - y[:, 2, :, :])).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
        loss = (lossu + lossv + lossp) / channels_weights
        return paddle.sum(loss), output


    # 训练模型，加入除loss以外的4个指标：Total MSE、Ux MSE、Uy MSE、p MSE
    DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(simulation_directory, model, loss_func,
                                                                              train_dataset, test_dataset, optimizer,
                                                                              epochs=epochs, batch_size=batch_size,
                                                                              m_mse_name="Total MSE",
                                                                              m_mse_on_batch=lambda scope: float(
                                                                                  paddle.sum((scope["output"] -
                                                                                              scope["batch"][1]) ** 2)),
                                                                              m_mse_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),
                                                                              m_ux_name="Ux MSE",
                                                                              m_ux_on_batch=lambda scope: float(
                                                                                  paddle.sum((scope["output"][:, 0, :,
                                                                                              :] - scope["batch"][1][:,
                                                                                                   0, :, :]) ** 2)),
                                                                              m_ux_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),
                                                                              m_uy_name="Uy MSE",
                                                                              m_uy_on_batch=lambda scope: float(
                                                                                  paddle.sum((scope["output"][:, 1, :,
                                                                                              :] - scope["batch"][1][:,
                                                                                                   1, :, :]) ** 2)),
                                                                              m_uy_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),
                                                                              m_p_name="p MSE",
                                                                              m_p_on_batch=lambda scope: float(
                                                                                  paddle.sum((scope["output"][:, 2, :,
                                                                                              :] - scope["batch"][1][:,
                                                                                                   2, :, :]) ** 2)),
                                                                              m_p_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]), patience=25,
                                                                              after_epoch=after_epoch
                                                                              )

    # 用于记录训练过程中的各项指标并保存
    metrics = {}
    metrics["train_metrics"] = train_metrics
    metrics["train_loss"] = train_loss
    metrics["test_metrics"] = test_metrics
    metrics["test_loss"] = test_loss
    curves = {}
    curves["train_loss_curve"] = train_loss_curve
    curves["test_loss_curve"] = test_loss_curve
    curves["train_mse_curve"] = train_mse_curve
    curves["test_mse_curve"] = test_mse_curve
    curves["train_ux_curve"] = train_ux_curve
    curves["test_ux_curve"] = test_ux_curve
    curves["train_uy_curve"] = train_uy_curve
    curves["test_uy_curve"] = test_uy_curve
    curves["train_p_curve"] = train_p_curve
    curves["test_p_curve"] = test_p_curve
    config["metrics"] = metrics
    config["curves"] = curves

    # 保存各项训练指标
    with open(simulation_directory + "results.json", "w") as file:
        json.dump(config, file)
