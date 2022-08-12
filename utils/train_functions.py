import copy
import paddle
import os

# 中间函数，按照读入的字典的键生成对应的列表字典
def generate_metrics_list(metrics_def):
    list = {}
    for name in metrics_def.keys():
        list[name] = []
    return list

# 训练最内层的循环，遍历数据集
def epoch(scope, loader, on_batch=None, training=False):
    model = scope["model"]
    optimizer = scope["optimizer"]
    loss_func = scope["loss_func"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)
    scope["loader"] = loader

    metrics_list = generate_metrics_list(metrics_def)
    total_loss = 0
    if training:
        model.train()
    else:
        model.eval()
    # 使用GPU进行训练
    with paddle.static.device_guard('gpu'):
        # 遍历数据集
        for tensors in loader:
            if "process_batch" in scope and scope["process_batch"] is not None:
                tensors = scope["process_batch"](tensors)
            if "device" in scope and scope["device"] is not None:
                tensors = [tensor.to(scope["device"]) for tensor in tensors]
            # 计算loss
            loss, output = loss_func(model, tensors)
            if training:
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            scope["batch"] = tensors
            scope["loss"] = loss
            scope["output"] = output
            scope["batch_metrics"] = {}
            for name, metric in metrics_def.items():
                value = metric["on_batch"](scope)
                scope["batch_metrics"][name] = value
                metrics_list[name].append(value)
            if on_batch is not None:
                on_batch(scope)
    scope["metrics_list"] = metrics_list
    metrics = {}
    for name in metrics_def.keys():
        scope["list"] = scope["metrics_list"][name]
        metrics[name] = metrics_def[name]["on_epoch"](scope)
    return total_loss, metrics

# 训练主循环
def train(save_path, scope, train_dataset, val_dataset, batch_size=256, print_function=print, eval_model=None,
          on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None):
    epochs = scope["epochs"]
    model = scope["model"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)

    scope["best_train_metric"] = None
    scope["best_train_loss"] = float("inf")
    scope["best_val_metrics"] = None
    scope["best_val_loss"] = float("inf")
    scope["best_model"] = None

    # 训练集随机打散（注：后面固定了随机种子，便于复现）
    train_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    val_sampler = paddle.io.DistributedBatchSampler(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 使用DataLoader加载训练集和测试集
    train_loader = paddle.io.DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = paddle.io.DataLoader(val_dataset, batch_sampler=val_sampler)

    skips = 0

    # 创建保存训练过程中的训练和测试结果的文件
    if os.path.isfile(os.path.join(save_path, "train_log.txt")):

        os.remove(os.path.join(save_path, "train_log.txt"))
        print_function("Previous train log deleted successfully")
    else:
        print_function("Train log does not exist")

    # 训练主循环
    for epoch_id in range(1, epochs + 1):
        scope["epoch"] = epoch_id
        # 保存训练过程中的训练和测试结果
        with open(os.path.join(save_path, "train_log.txt"), "a") as f:
            print_function("Epoch #" + str(epoch_id))
            f.write("Epoch #" + str(epoch_id) + "\n")
            scope["dataset"] = train_dataset
            # 模型在训练集中训练
            train_loss, train_metrics = epoch(scope, train_loader, on_train_batch, training=True)
            scope["train_loss"] = train_loss
            scope["train_metrics"] = train_metrics
            print_function("\tTrain Loss = " + str(train_loss))
            f.write("\tTrain Loss = " + str(train_loss) + "\n")
            for name in metrics_def.keys():
                print_function("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]))
                f.write("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]) + "\n")
            if on_train_epoch is not None:
                on_train_epoch(scope)
            del scope["dataset"]
            # 模型在测试集中验证
            scope["dataset"] = val_dataset
            with paddle.no_grad():
                val_loss, val_metrics = epoch(scope, val_loader, on_val_batch, training=False)
            scope["val_loss"] = val_loss
            scope["val_metrics"] = val_metrics
            print_function("\tValidation Loss = " + str(val_loss))
            f.write("\tValidation Loss = " + str(val_loss) + "\n")
            for name in metrics_def.keys():
                print_function("\tValidation " + metrics_def[name]["name"] + " = " + str(val_metrics[name]))
                f.write("\tValidation " + metrics_def[name]["name"] + " = " + str(val_metrics[name]) + "\n")
            if on_val_epoch is not None:
                on_val_epoch(scope)
            del scope["dataset"]
            # 按统计的loss确定最优模型，并保存
            is_best = None
            if eval_model is not None:
                is_best = eval_model(scope)
            if is_best is None:
                is_best = val_loss < scope["best_val_loss"]
            if is_best:
                scope["best_train_metric"] = train_metrics
                scope["best_train_loss"] = train_loss
                scope["best_val_metrics"] = val_metrics
                scope["best_val_loss"] = val_loss
                if epoch_id > 500:
                    paddle.save(model.state_dict(), os.path.join(save_path, "DeepCFD_" + str(epoch_id) + ".pdparams"))
                    print_function("Model saved!")
                    f.write("Model saved!" + "\n")
                skips = 0
            else:
                skips += 1
            if after_epoch is not None:
                after_epoch(scope)

    return scope["best_model"], scope["best_train_metric"], scope["best_train_loss"], \
           scope["best_val_metrics"], scope["best_val_loss"]

# 训练配置函数
def train_model(save_path, model, loss_func, train_dataset, val_dataset, optimizer, process_batch=None, eval_model=None,
                on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None,
                epochs=100, batch_size=256, **kwargs):
    scope = {}
    scope["model"] = model
    scope["loss_func"] = loss_func
    scope["train_dataset"] = train_dataset
    scope["val_dataset"] = val_dataset
    scope["optimizer"] = optimizer
    scope["process_batch"] = process_batch
    scope["epochs"] = epochs
    scope["batch_size"] = batch_size
    metrics_def = {}
    names = []
    for key in kwargs.keys():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "m":
            if parts[1] not in names:
                names.append(parts[1])
    for name in names:
        if "m_" + name + "_name" in kwargs and "m_" + name + "_on_batch" in kwargs and "m_" + name + "_on_epoch" in kwargs:
            metrics_def[name] = {
                "name": kwargs["m_" + name + "_name"],
                "on_batch": kwargs["m_" + name + "_on_batch"],
                "on_epoch": kwargs["m_" + name + "_on_epoch"],
            }
        else:
            print("Warning: " + name + " metric is incomplete!")
    scope["metrics_def"] = metrics_def
    return train(save_path, scope, train_dataset, val_dataset, eval_model=eval_model, on_train_batch=on_train_batch,
                 on_val_batch=on_val_batch, on_train_epoch=on_train_epoch, on_val_epoch=on_val_epoch,
                 after_epoch=after_epoch,
                 batch_size=batch_size)