from keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, GRU, Bidirectional, Flatten, Dense, Dropout
from callbacks import weighted_categorical_crossentropy

def build_model(input_shape, class_weights):
    """
    建立模型
    ---
    Build the model
    
    參數:
    input_shape：輸入數據的形狀。
    class_weights：每個類別的權重，用於加權損失函數。
    ---
    Parameters:
    input_shape: Shape of the input data.
    class_weights: Weights for each class, used for the weighted loss function.
    """
    input_tensor = Input(input_shape)
    x = input_tensor

    for i in range(3):
        x = Conv2D(32 * 2**i, (4, 4), activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32 * 2**i, (2, 2), activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

    x = Reshape((x.shape[1], -1))(x)

    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.1))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.1))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    outputs = [Dense(26, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]

    model = Model(inputs=input_tensor, outputs=outputs)

    model.compile(
        loss=[
            weighted_categorical_crossentropy(class_weights[0]),
            weighted_categorical_crossentropy(class_weights[1]),
            weighted_categorical_crossentropy(class_weights[2]),
            weighted_categorical_crossentropy(class_weights[3])
        ],
        optimizer='adam',
        metrics=['accuracy']
    )

    return model