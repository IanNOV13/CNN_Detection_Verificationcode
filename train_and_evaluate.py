import matplotlib.pyplot as plt
from callbacks import DelayedEarlyStopping,LearningRateScheduler,cosine_decay_schedule

def train_and_evaluate(model, train_data, val_split=0.2, batch_size=512, epochs=500, patience=100, start_epoch=50, fit_type='float32'):
    """
    開始訓練模型，並給出訓練報告。
    ---
    參數：
    model：建構的模型。
    train_data：訓練資料包含(標準化訓練特徵, 標準化測試特徵, 訓練標籤獨熱編碼, 測試標籤獨熱編碼, 各字符對於個字母的權重)。
    val_split：訓練時保留測適用的資料量，預設20%。
    batch_size：一次訓練的批量，預設512。
    epochs：總訓練次數，若達早停條件則會提前結束。
    patience：若loss沒有持續降低達指定次數就提早結束訓練，預設100次。
    start_epoch：模型預熱次數，前50次訓練不會早停。
    fit_type：訓練模型給予的資料格式。
    ---
    Start training the model and provide a training report.
    Parameters:

    model: The constructed model.
    train_data: Training data including (standardized training features, standardized test features, training labels one-hot encoding, test labels one-hot encoding, class weights for each character).
    val_split: The proportion of the training data to be used as validation data, default is 20%.
    batch_size: The batch size for each training iteration, default is 512.
    epochs: The total number of training iterations. Training will stop early if early stopping criteria are met.
    patience: If the loss does not decrease for a specified number of iterations, training will stop early. Default is 100 iterations.
    start_epoch: The number of initial epochs for model warm-up, early stopping will not be triggered during this period. Default is 50 epochs.
    fit_type: The data format provided to the training model.
    """
    train_feature_normalize, test_feature_normalize, train_label_onehot, test_label_onehot, class_weights = train_data

    delayed_early_stopping = DelayedEarlyStopping(monitor='val_loss', patience=patience, start_epoch=start_epoch)
    lr_scheduler = LearningRateScheduler(cosine_decay_schedule, verbose=1)

    try:
        model.load_weights(f'best_model_{fit_type}.h5')
        print(f"成功加載best_model_{fit_type}模型權重。")
    except Exception as e:
        print(f"加载模型权重失败:\n{str(e)}")

    train_history = model.fit(
                        train_feature_normalize,
                        [train_label_onehot[:, i] for i in range(4)], 
                        epochs=epochs,
                        batch_size=batch_size, 
                        validation_split=val_split,
                        callbacks=[delayed_early_stopping, lr_scheduler],
                        verbose=2,
                    )

    scores = model.evaluate(test_feature_normalize, [test_label_onehot[:, i] for i in range(4)])
    total_loss = scores[0]
    c1_loss, c1_accuracy = scores[1:3]
    c2_loss, c2_accuracy = scores[3:5]
    c3_loss, c3_accuracy = scores[5:7]
    c4_loss, c4_accuracy = scores[7:9]
    
    print(f"總損失 = {total_loss}")
    print(f"準確度 = {c1_accuracy}, {c2_accuracy}, {c3_accuracy}, {c4_accuracy}")

    model.save(f'best_model_{fit_type}.h5')
    print(f"成功保存模型到 best_model_{fit_type}.h5。")

    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    for i in range(4):
        plt.plot(train_history.history[f'c{i+1}_accuracy'], label=f'Training Accuracy (c{i+1})')
        plt.plot(train_history.history[f'val_c{i+1}_accuracy'], label=f'Validation Accuracy (c{i+1})')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
