# 本文件用于评估模型指标
import pickle
from utils.train_functions import *
from utils.functions import *
from model.UNetEx import UNetEx
import configparser

if __name__ == "__main__":
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

    # 按7：3的比例分割数据集，7为训练集，3为测试集
    train_data, test_data = split_tensors(x, y, ratio=float(config["hyperparameter"]["train_test_ratio"]))

    train_dataset, test_dataset = paddle.io.TensorDataset([train_data[0], train_data[1]]), \
                                  paddle.io.TensorDataset([test_data[0], test_data[1]])
    test_x, test_y = test_dataset[:]

    # 设置卷积核大小
    kernel_size = int(config["net_parameter"]["kernel_size"])
    # 设置卷积层channel数目
    filters = [int(i) for i in config["net_parameter"]["filters"].split(",")]
    # 设置batch_norm和weight_norm
    bn = bool(int(config["net_parameter"]["batch_norm"]))
    wn = bool(int(config["net_parameter"]["weight_norm"]))
    # 构建模型
    model = UNetEx(3, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    # 加载模型参数
    model.set_state_dict(
        paddle.load(os.path.join(config["path"]["save_path"], config["path"]["model_name"])))
    # 测试训练模型
    out = model(test_x)
    # 指标符合性分析
    # Total MSE
    Total_MSE = paddle.sum((out - test_y) ** 2) / len(test_x)
    # Ux MSE
    Ux_MSE = paddle.sum((out[:, 0, :, :] - test_y[:, 0, :, :]) ** 2) / len(test_x)
    # Uy MSE
    Uy_MSE = paddle.sum((out[:, 1, :, :] - test_y[:, 1, :, :]) ** 2) / len(test_x)
    # p MSE
    p_MSE = paddle.sum((out[:, 2, :, :] - test_y[:, 2, :, :]) ** 2) / len(test_x)
    print("Total MSE is {}, Ux MSE is {}, Uy MSE is {}, p MSE is {}".format(Total_MSE.detach().numpy()[0],
                                                                            Ux_MSE.detach().numpy()[0],
                                                                            Uy_MSE.detach().numpy()[0],
                                                                            p_MSE.detach().numpy()[0]))
